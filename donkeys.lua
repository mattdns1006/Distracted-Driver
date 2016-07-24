Threads = require "threads"
dofile("display.lua")

function newEpoch()
	do 
		local threadParams = params
		donkeys = Threads(
				params.nThreads,
				function(idx)
					tid = idx
					dofile("provider.lua")
					params = threadParams
					prov = Provider.new(tid,params.nThreads,params.cv)
					--print(string.format("Initialized thread %d of %d.", tid,params.nThreads))
				end
				)
	end

end

function reset()
	local nThreadsResetted = Counter.new()
	while true do 
	donkeys:addjob(
		function ()

			prov.trainData.currentIdx = 1
			prov.testData.currentIdx = 1
			prov.trainData.finished = 0
			prov.testData.finished = 0
			return tid
		end,
		function (tid)
			nThreadsResetted:add(tid)
		end
		)
		if nThreadsResetted:size() == params.nThreads then 
			break
		end

	end
end

function train()
	local count = 0
	local nThreadsFinished = 0 
	optimState = {
		learningRate = params.lr,
		beta1 = 0.9,
		beta2 = 0.999,
		epsilon = 1e-8
	}
	local epochLosses = {}
	reset()
	while true do
		donkeys:addjob(function()
					imgPaths, inputs, targets = prov:getBatch("train") 
					return tid, prov.trainData.finished, imgPaths,inputs,targets
			       end,
			       function(tid, finished, imgPaths, inputs, targets)

				       if finished <= 1 then
					       if finished == 1 then nThreadsFinished = nThreadsFinished + 1; print(string.format("Thread %d finished training (total = %d)",tid,nThreadsFinished))
					       end
					        if parameters == nil then 
							if model then parameters, gradParameters = model:getParameters() end
							print("Number of parameters ==>",parameters:size())
						end
						local outputs
						local dLoss_dO
						local batchLoss
						local targetResize
						local loss
						function feval(x)
							if x ~= parameters then parameters:copy(x) end
							model:training()
							gradParameters:zero()
							outputs = model:forward(inputs) -- Only one input for training unlike testing
							loss = criterion:forward(outputs,targets)
							dLoss_dO = criterion:backward(outputs,targets)
							model:backward(inputs,dLoss_dO)
							return	loss, gradParameters 
						end

						_, batchLoss = optimMethod(feval,parameters,optimState)
						epochLosses[#epochLosses+1] = loss
					       count = targets:size(1) + count
					       xlua.progress(count,dataSizes.trainCV)
					       cmTrain:batchAdd(outputs,targets)
					       if torch.uniform() < 0.1 and params.display == 1 then display(inputs,"train") end
					elseif finished == 2 then 
					end
				end
				)
				if nThreadsFinished == params.nThreads then break end 
				donkeys:synchronize()
				
	end
	local meanLoss = torch.Tensor(epochLosses):mean()
	print(string.format("Finished training epoch %d with %d examples seen. Mean loss = %f.",epoch,count,meanLoss))
	print(cmTrain)
	cmTrain:zero()
end

function test()
	local count = 0
	local nThreadsFinished = 0 
	local epochLosses = {}
	while true do
		donkeys:addjob(function()
					imgPaths, inputs, targets = prov:getBatch("test") 
					return tid, prov.testData.finished, imgPaths,inputs,targets
			       end,
			       function(tid, finished, imgPaths, inputs, targets)
				       if finished <= 1 then
					       if finished == 1 then nThreadsFinished = nThreadsFinished + 1; print(string.format("Thread %d finished testing (total = %d)",tid,nThreadsFinished))
					       end
					local outputs
					local loss
					local targetResize

					outputs = model:forward(inputs)
					loss = criterion:forward(outputs,targets)
				        count = targets:size(1) + count
				        xlua.progress(count,dataSizes.testCV)
					cmTest:batchAdd(outputs,targets)
					epochLosses[#epochLosses+1] = loss
					if torch.uniform() < 0.1 and params.display == 1 then display(inputs,"test") end
					elseif finished == 2 then 
					end
				end
			)
			if nThreadsFinished == params.nThreads then break end 
			donkeys:synchronize()
	end
	local meanLoss = torch.Tensor(epochLosses):mean()
	print(string.format("Finished test epoch %d with %d examples seen. Mean loss = %f.",epoch,count,meanLoss))
	print(cmTest)
	cmTest:zero()
end
	
