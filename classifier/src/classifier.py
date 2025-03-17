# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

import torch
import torch.nn.functional as F

class ResBlock3d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels_conv, **kwargs):
        super().__init__()
        if hidden_channels_conv<=0:
            hidden_channels = min(in_channels, out_channels)
        else:
            hidden_channels = hidden_channels_conv
        self.layers = torch.nn.Sequential(
            torch.nn.BatchNorm3d(in_channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels, hidden_channels, **kwargs),
            torch.nn.BatchNorm3d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(hidden_channels, out_channels, **kwargs)
        )

        if in_channels != out_channels:
            self.skip = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x):
        return self.layers(x) + self.skip(x)

class ResnetOld(torch.nn.Module):
    def __init__(self, channels, output_channels):
        super().__init__()
        self.skip_block = torch.nn.Conv3d(channels, channels, kernel_size=1)
        self.right_block = ResBlock3d(channels, output_channels, 0, kernel_size=1)

    def forward(self,x):
        x = x + self.skip_block(x)
        x = self.right_block(x)
        return x
   
def get_classifier(args, embedding_inference):
    # return torch.nn.Sequential(torch.nn.Conv3d(args.embedding_size,1024,1), torch.nn.ReLU(), ToDouble(), torch.nn.Conv3d(1024,1,1).double(), ToFloat())
    # return torch.nn.Sequential(torch.nn.InstanceNorm3d(args.embedding_size), torch.nn.Conv3d(args.embedding_size,1024,1), torch.nn.ReLU(), ToDouble(), torch.nn.Conv3d(1024,1,1).double(), ToFloat())
    
    classifier = torch.nn.ModuleList()
    upconv = torch.nn.ModuleList()
    skip_blocks = torch.nn.ModuleList()
    if isinstance(args.embedding_size, (list,tuple)):
        current_x_size = args.embedding_size[-1]
    else:
        current_x_size = args.embedding_size
    for embedding_level in range(max(args.embedding_levels-2,0),-1,-1):
        if isinstance(args.embedding_size, (list,tuple)):
            channels = args.embedding_size[embedding_level]
        else:
            channels = args.embedding_size
        if embedding_level==0:
            from global_ import finding_types
            if args.normal_abnormal is not None:
                output_channels = 1
            else:
                output_channels = len(finding_types)
        else:
            output_channels = channels
        if 'conv' in args.upsampling_classifier:
            upconv.append(torch.nn.Conv3d(current_x_size, channels, kernel_size=1))
            skip_blocks.append(torch.nn.Conv3d(channels, channels, kernel_size=1))
            current_x_size = channels
        if args.join_levels_classifier in ['add', 'sum']:
            assert(current_x_size==channels)
        elif args.join_levels_classifier in ['cat', 'concat', 'concatenate']:
            current_x_size = current_x_size + channels
        if embedding_level>0 and args.concat_type_classifier=='up':
            classifier.append(torch.nn.Identity())
        else:
            classifier.append(torch.nn.Sequential(ResBlock3d(current_x_size, current_x_size, args.hidden_channels_conv, kernel_size=1),
                                                    torch.nn.Dropout(p=args.dropout_p),
                                                    ResBlock3d(current_x_size, output_channels, args.hidden_channels_conv, kernel_size=1)))

            current_x_size = output_channels
        
    classifier = classifier[::-1]
    upconv = upconv[::-1]
    skip_blocks = skip_blocks[::-1]
    if args.join_levels_classifier in ['add', 'sum']:
        join = lambda x,y: x+y
    elif args.join_levels_classifier in ['cat', 'concat', 'concatenate']:
        join = lambda x,y: torch.cat((x,y),dim = 1)
    if 'nearest' in args.upsampling_classifier:
        print(args.upsampling_classifier)
        def up(x, destination_shape):
            # print(x.shape, destination_shape)
            return F.interpolate(x, destination_shape, mode='nearest')
        
        # up = lambda x, destination_shape: torch.tensor(skimage.transform.resize(x[0][0].numpy(), destination_shape, order = 0)[None][None])
    elif 'linear' in args.upsampling_classifier:
        up = lambda x, destination_shape: F.interpolate(x, destination_shape, mode='trilinear', align_corners=False)
    class JoinModels(torch.nn.Module):
        def __init__(self,embedding_model_,classifier, upconv, skip_blocks):   
            super().__init__()
            self.embedding_model = embedding_model_
            self.classifier = classifier
            self.upconv = upconv
            self.skip_blocks = skip_blocks

        def forward(self, input_image):
            x = input_image['image']
            original_x_shape = x.shape
            x = x.view([-1,x.shape[2]])
            x = self.classifier[0](x[:,:,None,None,None])[:,:,0,0,0]
            x = x.view([original_x_shape[0], original_x_shape[1], -1])
            return x
            to_return = []
            # for index_batch in range(len(input_image['segmentation'])):
            #     embedded = embedding_inference({'img_metas':[input_image['img_metas'][index_batch]]}, self.embedding_model)
            #     if not isinstance(embedded, (list, tuple)):
            #         embedded = [embedded]
            #     x = embedded[-1].detach()
                
            #     for i in range(max(len(embedded)-2,0),-1,-1):
            #         # print(embedded)
            #         if i != len(embedded)-1:
            #             if 'conv' in args.upsampling_classifier:
            #                 x = self.upconv[i](x)
            #             x = up(x, embedded[i].shape[-3:])
            #             if args.normalize_coarse:
            #                 x = F.normalize(x, dim=1)
            #             if 'conv' in args.upsampling_classifier:
            #                 x = join(self.skip_blocks[i](embedded[i].detach()), x)
            #             else:
            #                 x = join(embedded[i].detach(), x)
                    
            #         x = x.float()

            #         if not isinstance(self.classifier, (list, tuple, torch.nn.ModuleList)):
            #             x = self.classifier(x)
            #         else:
            #             x = self.classifier[i](x)
            #     del embedded
            #     x = masked_avg_pool2d(x[:,None],input_image['segmentation'][index_batch][:,None])
            #     to_return.append(x)

            return torch.cat(to_return, axis = 0)
    return classifier, upconv, skip_blocks, JoinModels
