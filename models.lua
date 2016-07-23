require "nn"
require "nngraph"

local Convolution = nn.SpatialConvolution
local Pool = nn.SpatialMaxPooling
local fmp = nn.SpatialFractionalMaxPooling
local UpSample = nn.SpatialUpSamplingNearest
local SBN = nn.SpatialBatchNormalization
local Dropout = nn.Dropout
local af = nn.ReLU
local Linear = nn.Linear
local Dropout = nn.Dropout
local layers = dofile("/home/msmith/torchFunctions/layers.lua")

models = {}

function initParamsEg()
	params = {}
	params.kernelSize = 3
	params.nFeats = 22
	params.nDown = 7
	params.nUp = 3 
	model = nn.Sequential()
end
--initParamsEg()

local nFeats = params.nFeats 
local nFeatsInc = torch.floor(params.nFeats/4)
local nOutputs
local nInputs
local kS = 3 
local pad = torch.floor((kS-1)/2)

function shortcut(nInputPlane, nOutputPlane, stride)
	return nn.Sequential()
		:add(Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0))
		:add(SBN(nOutputPlane))
		--:add(nn.Identity())
end
	
function basicblock(nInputPlane, n, stride)
	local s = nn.Sequential()

	s:add(Convolution(nInputPlane,n,3,3,1,1,1,1))
	s:add(SBN(n))
	s:add(af())

	return nn.Sequential()
	 :add(nn.ConcatTable()
	    :add(s)
	    :add(shortcut(nInputPlane, n, stride)))
	 :add(nn.CAddTable(true))
	 :add(af())

end

function models.model1()
	local model = nn.Sequential()
	local nInputs
	local nOutputs
	for i =1, 8 do
		if i == 1 then nInputs = 3; else nInputs = nOutputs; end
		if i == 1 then nOutputs = nFeats; else nOutputs = nOutputs; end
		model:add(basicblock(nInputs,nOutputs,1))
		model:add(fmp(2,2,0.8,0.8))
		model:add(Dropout(0.4))
	end
	model:add(Convolution(nInputs,1,3,3,1,1,1,1))
	nInputs = nOutputs
	outputBeforeReshape = model:cuda():forward(torch.rand(1,3,params.inH,params.inW):cuda()):size()
	nOutputsBeforeReshape = outputBeforeReshape[2]*outputBeforeReshape[3]*outputBeforeReshape[4]
	model:add(nn.Reshape(nOutputsBeforeReshape))
	model:add(nn.Linear(nOutputsBeforeReshape,200))
	model:add(nn.BatchNormalization(200))
	model:add(af())
	model:add(Dropout(0.5))
	model:add(nn.Linear(200,10))
	layers.init(model)
	return model
end

return models
