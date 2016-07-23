Threads = require "threads"

do 
	local threadParams = params
	slaves = Threads(
			params.nThreads,
			function(idx)
				params = threadParams
				require "xlua"
				require "string"
				require "image"
				require "cunn"

				tid = idx -- Thread id
				print(string.format("Initialized thread %d of %d.", tid,params.nThreads))
				loadData = require "loadData"
				trainPaths, testPaths, allTrainPaths = loadData.init(tid,params.nThreads,params.cv)
			end
			)
end
	
