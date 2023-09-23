import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import  Tensor


class DropBlock(nn.Module):
    def __init__(self, block_size: int = 5, p: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: Tensor) -> float:
        

        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.size()
        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


class double_conv(nn.Module):
  def __init__(self,intc,outc):
    super().__init__()
    self.conv1=nn.Conv2d(intc,outc,kernel_size=3,padding=1,stride=1)
    self.drop1= DropBlock(7, 0.9)
    self.bn1=nn.BatchNorm2d(outc)
    self.relu1=nn.ReLU()
    self.conv2=nn.Conv2d(outc,outc,kernel_size=3,padding=1,stride=1)
    self.drop2=DropBlock(7, 0.9)
    self.bn2=nn.BatchNorm2d(outc)
    self.relu2=nn.ReLU()

  def forward(self,input):
    x=self.relu1(self.bn1(self.drop1(self.conv1(input))))
    x=self.relu2(self.bn2(self.drop2(self.conv2(x))))

    return x
class upconv(nn.Module):
  def __init__(self,intc,outc) -> None:
    super().__init__()
    self.up=nn.ConvTranspose2d(intc, outc, kernel_size=2, stride=2, padding=0)
   # self.relu=nn.ReLU()

  def forward(self,input):
    x=self.up(input)
    #x=self.relu(self.up(input))
    return x
class unet(nn.Module):
  def __init__(self,int,out) -> None:
    'int: represent the number of image channels'
    'out: number of the desired final channels'

    super().__init__()
    'encoder'
    self.convlayer1=double_conv(int,64)
    self.down1=nn.MaxPool2d((2, 2))
    self.convlayer2=double_conv(64,128)
    self.down2=nn.MaxPool2d((2, 2))
    self.convlayer3=double_conv(128,256)
    self.down3=nn.MaxPool2d((2, 2))
    self.convlayer4=double_conv(256,512)
    self.down4=nn.MaxPool2d((2, 2))

    'bridge'
    self.bridge=double_conv(512,1024)
    'decoder'
    self.up1=upconv(1024,512)
    self.convlayer5=double_conv(1024,512)
    self.up2=upconv(512,256)
    self.convlayer6=double_conv(512,256)
    self.up3=upconv(256,128)
    self.convlayer7=double_conv(256,128)
    self.up4=upconv(128,64)
    self.convlayer8=double_conv(128,64)
    'output'
    self.outputs = nn.Conv2d(64, out, kernel_size=1, padding=0)
    self.sig=nn.Sigmoid()
  def forward(self,input):
    'encoder'
    l1=self.convlayer1(input)
    d1=self.down1(l1)
    l2=self.convlayer2(d1)
    d2=self.down2(l2)
    l3=self.convlayer3(d2)
    d3=self.down3(l3)
    l4=self.convlayer4(d3)
    d4=self.down4(l4)
    'bridge'
    bridge=self.bridge(d4)
    'decoder'
    up1=self.up1(bridge)
    up1 = torch.cat([up1, l4], axis=1)
    l5=self.convlayer5(up1)

    up2=self.up2(l5)
    up2 = torch.cat([up2, l3], axis=1)
    l6=self.convlayer6(up2)

    up3=self.up3(l6)
    up3= torch.cat([up3, l2], axis=1)
    l7=self.convlayer7(up3)

    up4=self.up4(l7)
    up4 = torch.cat([up4, l1], axis=1)
    l8=self.convlayer8(up4)
    out=self.outputs(l8)

    #out=self.sig(self.outputs(l8))
    return out
class spatialAttention(nn.Module):
  def __init__(self) -> None:
    super().__init__()

    self.conv77=nn.Conv2d(2,1,kernel_size=7,padding=3)
    self.sig=nn.Sigmoid()
  def forward(self,input):
    x=torch.cat( (torch.max(input,1)[0].unsqueeze(1), torch.mean(input,1).unsqueeze(1)), dim=1 )
    x=self.sig(self.conv77(x))
    x=input*x
    return x
class SAunet(nn.Module):
  def __init__(self,int,out) -> None:
    'int: represent the number of image channels'
    'out: number of the desired final channels'

    super().__init__()
    'encoder'
    self.convlayer1=double_conv(int,64)
    self.down1=nn.MaxPool2d((2, 2))
    self.convlayer2=double_conv(64,128)
    self.down2=nn.MaxPool2d((2, 2))
    self.convlayer3=double_conv(128,256)
    self.down3=nn.MaxPool2d((2, 2))
    self.convlayer4=double_conv(256,512)
    self.down4=nn.MaxPool2d((2, 2))

    'bridge'
    self.attmodule=spatialAttention()
    self.bridge1=nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=1)
    self.bn1=nn.BatchNorm2d(1024)
    self.act1=nn.ReLU()
    self.bridge2=nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=1)
    self.bn2=nn.BatchNorm2d(1024)
    self.act2=nn.ReLU()
    'decoder'
    self.up1=upconv(1024,512)
    self.convlayer5=double_conv(1024,512)
    self.up2=upconv(512,256)
    self.convlayer6=double_conv(512,256)
    self.up3=upconv(256,128)
    self.convlayer7=double_conv(256,128)
    self.up4=upconv(128,64)
    self.convlayer8=double_conv(128,64)
    'output'
    self.outputs = nn.Conv2d(64, out, kernel_size=1, padding=0)
    self.sig=nn.Sigmoid()
  def forward(self,input):
    'encoder'
    l1=self.convlayer1(input)
    d1=self.down1(l1)
    l2=self.convlayer2(d1)
    d2=self.down2(l2)
    l3=self.convlayer3(d2)
    d3=self.down3(l3)
    l4=self.convlayer4(d3)
    d4=self.down4(l4)
    'bridge'
    b1=self.act1(self.bn1(self.bridge1(d4)))
    att=self.attmodule(b1)
    b2=self.act2(self.bn2(self.bridge2(att)))
    'decoder'
    up1=self.up1(b2)
    up1 = torch.cat([up1, l4], axis=1)
    l5=self.convlayer5(up1)

    up2=self.up2(l5)
    up2 = torch.cat([up2, l3], axis=1)
    l6=self.convlayer6(up2)

    up3=self.up3(l6)
    up3= torch.cat([up3, l2], axis=1)
    l7=self.convlayer7(up3)

    up4=self.up4(l7)
    up4 = torch.cat([up4, l1], axis=1)
    l8=self.convlayer8(up4)
    out=self.outputs(l8)

    #out=self.sig(self.outputs(l8))
    return out
class attentionGate(nn.Module):
  def __init__(self,fg,fx,fint) :
      super().__init__()
      self.gatingSigconv=nn.Conv2d(fg, fg//2, kernel_size=3, padding=1)
      self.bng = nn.BatchNorm2d(fint)
      self.skipConnexion=nn.Conv2d(fx, fg//2, kernel_size=3,stride=2, padding=1)
      self.bnskip = nn.BatchNorm2d(fint)
      self.phi=nn.Conv2d(fg//2, 1, kernel_size=3, padding=1)
      self.sig=nn.Sigmoid()
      self.relu=nn.ReLU()
      self.resampe=nn.Upsample(scale_factor=2)
  def forward(self ,skip,gate):

    intm=self.relu(self.skipConnexion(skip)+self.gatingSigconv(gate))

    #make attentin coifeicent betzeen 0 and 1
    intm=self.sig(self.phi(intm))
    #resample the attention matrix
    intm=self.resampe(intm)
    xi=skip*intm
    return xi

class attunet(nn.Module):
  def __init__(self,intc,out) -> None:
    'int: represent the number of image channels'
    'out: number of the desired final channels'

    super().__init__()
    'encoder'
    self.convlayer1=double_conv(intc,64)
    self.down1=nn.MaxPool2d((2, 2))
    self.convlayer2=double_conv(64,128)
    self.down2=nn.MaxPool2d((2, 2))
    self.convlayer3=double_conv(128,256)
    self.down3=nn.MaxPool2d((2, 2))
    self.convlayer4=double_conv(256,512)
    self.down4=nn.MaxPool2d((2, 2))

    'bridge'
    self.bridge=double_conv(512,1024)
    'decoder'
    self.up1=upconv(1024,512)
    self.convlayer5=double_conv(1024,512)
    self.up2=upconv(512,256)
    self.convlayer6=double_conv(512,256)
    self.up3=upconv(256,128)
    self.convlayer7=double_conv(256,128)
    self.up4=upconv(128,64)
    self.convlayer8=double_conv(128,64)
    'output'
    self.outputs = nn.Conv2d(64, out, kernel_size=1, padding=0)
    self.sig=nn.Sigmoid()
    'attention modules'
    self.attgate1=attentionGate(1024,512,1024)
    self.attgate2=attentionGate(512,256,512)
    self.attgate3=attentionGate(256,128,256)
    self.attgate4=attentionGate(128,64,128)
  def forward(self,input):
    'encoder'
    l1=self.convlayer1(input)
    d1=self.down1(l1)
    l2=self.convlayer2(d1)
    d2=self.down2(l2)
    l3=self.convlayer3(d2)
    d3=self.down3(l3)
    l4=self.convlayer4(d3)
    d4=self.down4(l4)
    'bridge'
    bridge=self.bridge(d4)
    'decoder'
    l4=self.attgate1(l4,bridge)
    up1=self.up1(bridge)
    up1 = torch.cat([up1, l4], axis=1)
    l5=self.convlayer5(up1)

    l3=self.attgate2(l3,l5)

    up2=self.up2(l5)
    up2 = torch.cat([up2, l3], axis=1)
    l6=self.convlayer6(up2)


    l2=self.attgate3(l2,l6)
    up3=self.up3(l6)
    up3= torch.cat([up3, l2], axis=1)
    l7=self.convlayer7(up3)
    l1=self.attgate4(l1,l7)

    up4=self.up4(l7)
    up4 = torch.cat([up4, l1], axis=1)
    l8=self.convlayer8(up4)
    out=self.outputs(l8)

    #out=self.sig(self.outputs(l8))
    return out
