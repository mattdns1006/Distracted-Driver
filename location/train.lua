function slaveTrain()
		slaves:addjob(function()
				       x,y = loadData.getData("train",trainPaths,0)
				       return x,y
			       end,
			       function(x,y)
				        local yPred,loss
					model:training()
					yPred, loss = train(x,y)
					trainLosses:add(loss)
					if trainIdx % 500 == 0 and trainIdx > params.ma then
						print("Observations = ",obs)
						print("Moving average ("..params.ma..") loss = ",trainLosses:getLast())
						if params.display == 1 then display(x,y,yPred,"train") end
					end
				end
			)
end


function train(X,Y)

	--local yPred 
	local dLoss_dO
	local loss
	local yPred

	if trainIdx == nil then
		trainIdx = 1
		if model then parameters,gradParameters = model:getParameters() end
		print("Number of parameters ==>")
		print(parameters:size())
		optimState = {
			learningRate = params.lr,
			beta1 = 0.9,
			beta2 = 0.999,
			epsilon = 1e-8
		}

		optimMethod = optim.adam
	end
	
	function feval(x)
		if x ~= parameters then parameters:copy(x) end
		
		gradParameters:zero()
		yPred = model:forward(X)
		loss = criterion:forward(yPred,Y)
		dLoss_dO = criterion:backward(yPred,Y)
		model:backward(X,dLoss_dO)

		return	loss, gradParameters 
	end

	_, _ = optimMethod(feval,parameters,optimState)

	trainIdx = trainIdx + 1 
	--xlua.progress(trainIdx,350)
	collectgarbage()
	return yPred,loss, trainIdx

end


