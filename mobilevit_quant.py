from transformers import MobileViTForImageClassification
import torch
import torch.nn as nn
from models.ptq import QConv2d, QAct, QLinear, QIntLayerNorm, QIntSoftmax
from functools import partial

class QMobileViT(nn.Module):
	def __init__(self, model, cfg):
		super().__init__()
		self.model = model
		modules = []
		for name, m in model.named_modules():
			modules.append((name, m))
		for name, m in modules:
			if isinstance(m, nn.Conv2d):
				bool_bias = m.bias is not None
				quant_m = QConv2d(
					in_channels=m.in_channels,
					out_channels=m.out_channels,
					kernel_size=m.kernel_size,
					stride=m.stride,
					padding=m.padding,
					dilation=m.dilation,
					groups=m.groups,
					bias=bool_bias,
					quant=False,
					calibrate=False,
					bit_type=cfg.BIT_TYPE_W,
					calibration_mode=cfg.CALIBRATION_MODE_W,
					observer_str=cfg.OBSERVER_W,
					quantizer_str=cfg.QUANTIZER_W
				)
				quant_m.weight.data = m.weight.data.clone()
				if m.bias is not None:
					quant_m.bias.data = m.bias.data.clone()
				module_path = name.split('.')
				module = self.model
				for i, attr in enumerate(module_path):
					if i == len(module_path) - 1:
						setattr(module, attr, quant_m)
					else:
						module = getattr(module, attr)
			elif isinstance(m, nn.Linear):
				bool_bias = m.bias is not None
				quant_m = QLinear(
					in_features=m.in_features,
					out_features=m.out_features,
					bias=bool_bias,
					quant=False,
					calibrate=False,
					bit_type=cfg.BIT_TYPE_W,
					calibration_mode=cfg.CALIBRATION_MODE_W,
					observer_str=cfg.OBSERVER_W,
					quantizer_str=cfg.QUANTIZER_W
				)
				quant_m.weight.data = m.weight.data.clone()
				if m.bias is not None:
					quant_m.bias.data = m.bias.data.clone()
				module_path = name.split('.')
				module = self.model
				for i, attr in enumerate(module_path):
					if i == len(module_path) - 1:
						setattr(module, attr, quant_m)
					else:
						module = getattr(module, attr)
		self.cfg = cfg
	
	def forward(self, x):
		x = self.model(x)
		return x

	def get_quantized_model(self, cfg):
		modules = []
		for name, m in self.model.named_modules():
			modules.append((name, m))
		for name, m in modules:
			if isinstance(m, QConv2d):
				quant_m = m
				quant_m.weight.data = m.quantizer.quant(m.weight.data.clone()).type(torch.int8)
				if m.bias is not None:
					quant_m.bias.data = m.bias.data.clone()
				module_path = name.split('.')
				module = self.model
				for i, attr in enumerate(module_path):
					if i == len(module_path) - 1:
						setattr(module, attr, quant_m)
					else:
						module = getattr(module, attr)
			elif isinstance(m, QLinear):
				quant_m = m
				quant_m.weight.data = m.quantizer.quant(m.weight.data.clone()).type(torch.int8)
				if m.bias is not None:
					quant_m.bias.data = m.bias.data.clone()
				module_path = name.split('.')
				module = self.model
				for i, attr in enumerate(module_path):
					if i == len(module_path) - 1:
						setattr(module, attr, quant_m)
					else:
						module = getattr(module, attr)

	def model_quant(self):
		for m in self.model.modules():
			if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
				m.quant = True
			if self.cfg.INT_NORM:
				if type(m) in [QIntLayerNorm]:
					m.mode = 'int'

	def model_dequant(self):
		for m in self.model.modules():
			if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
				m.quant = False

	def model_open_calibrate(self):
		for m in self.model.modules():
			if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
				m.calibrate = True

	def model_open_last_calibrate(self):
		for m in self.model.modules():
			if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
				m.last_calibrate = True

	def model_close_calibrate(self):
		for m in self.model.modules():
			if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
				m.calibrate = False
