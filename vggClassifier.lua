----------------------------------------------------------------------
-- This script shows how to train different models on the MNIST 
-- dataset, using multiple optimization techniques (SGD, LBFGS)
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'pl'
require 'paths'
local DataLoader = require 'dataloader'

----------------------------------------------------------------------
-- parse command-line options
--
opt = {
    dataset = 'imagenet',       -- imagenet / lsun / folder
    data = 'data',              -- path to your train and val folders
    batchSize = 64,
    loadSize = 96,
    fineSize = 64,
    nThreads = 4,           -- #  of data loading threads to use
    plot = 1,               -- if 1 then plot, else not
    niter = 25,             -- #  of iter at starting learning rate
    ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
    display = 0,            -- display samples while training. 0 = false
    display_id = 20,        -- display window id.
    gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
    name = 'experiment-compositionalMNIST',
    ip='172.27.21.146',     -- the ip for display
    port=8000,        -- the port for display
    save_freq=5,      -- the frequency with which the parameters are saved
    s = "logs",       -- subdirectory to save logs
    n = "",           -- reload pretrained network
    lr = 0.0002,      -- initial learning rate for adam
    beta1 = 0.5,      -- momentum term of adam
    nc = 3,
    ndf = 64,         -- number of discrete filters in first conv layer
    manualSeed = 1,
    tenCrop = false,  -- Ten-crop testing
    tensorType = 'torch.CudaTensor',
    gen = 'gen',       -- Path to save generated files
    type='cuda',
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end
-- fix seed
torch.manualSeed(opt.manualSeed)

-- threads
torch.setnumthreads(opt.nThreads) -- TODO: why is it 1 in main code
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats, for SGD
torch.setdefaulttensortype('torch.FloatTensor')

local trainLoader, valLoader = DataLoader.create(opt)
print("Train Dataset: " .. opt.dataset, " Size: ", trainLoader:size())
print("Val Dataset: " .. opt.dataset, " Size: ", valLoader:size())

local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') then
        m.weight:normal(0.0, 0.02)
        m:noBias()
    elseif name:find('BatchNormalization') then
        if m.weight then m.weight:normal(1.0, 0.02) end
        if m.bias then m.bias:fill(0) end
    end
end
----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
local nclasses=1000
classes = {}
for i = 1,nclasses do
   classes[i] = tostring(i)
end

-- geometry: width and height of input images
geometry = {32,32}

local nc = opt.nc
local ndf = opt.ndf

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution

-- define model to train
------------------------------------------------------------
-- convolutional network 
------------------------------------------------------------


local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end

--print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
--model:add(nn.BatchFlip():float())
model:add(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor()))))
model:add(dofile('vgg.lua'))
model:get(2).updateGradInput = function(input) return end


--model=dofile('vgg.lua')
-- loss function: negative log-likelihood
--
local criterion = nn.CrossEntropyCriterion()

optimState = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
}

local input = torch.Tensor(opt.batchSize, 3, geometry[1], geometry[2])
local label = torch.Tensor(opt.batchSize)
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------
-- get/create dataset
nbTrainingPatches = 240000
nbTestingPatches = 40000

if opt.gpu > 0 then
    require 'cunn'
    cutorch.setDevice(opt.gpu)
    model:cuda();                   
    criterion:cuda();
    input=input:cuda();
    label=label:cuda();
end
-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<mnist> using model:')
print(model)

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.s, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.s, 'test.log'))
local err=0
local feval = function(x)
   -- just in case:
   collectgarbage()
   -- get new parameters
   -- reset gradients
   gradParameters:zero()
   data_tm:reset(); data_tm:resume()
  -- evaluate function for complete mini batch
   data_tm:stop()
   local output = model:forward(input)
   err = criterion:forward(output, label)
   -- estimate df/dW
   local df_do = criterion:backward(output, label)
   model:backward(input, df_do)
   -- update confusion
--   print(output:size())
   local dummy, pred=output:max(2)
   --print(label)
   local num_correct=0
   for i = 1,opt.batchSize do    --Todo: Can we change the iteration to the size of real. Is real always opt.batchsize big? It is in test() as well
      confusion:add(output[i], label[i]) 
      if pred[i]==label[i] then 
          num_correct=1+num_correct
      end
   end
   --local acc=num_correct
  -- print('Accuracy '.. acc)
   -- return f and df/dX
   return err,gradParameters
end

-- training function
function train(dataloader)
   -- epoch tracker
   epoch = epoch or 1
   -- local vars
   epoch_tm:reset()
   local trainSize = dataloader:size()-- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for n, sample in dataloader:run() do
      tm:reset()
      input:resize(sample.input:size()):copy(sample.input:cuda())
      label:resize(sample.target:size()):copy(sample.target:cuda())      --sample:copy(sam)
      optim.adam(feval, parameters, optimState)
      -- create mini batch
      print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
      .. '  Err: %.4f'):format(
      epoch, n, trainSize,
      tm:time().real, data_tm:time().real,
      err and err or -1))
--      print("Accuracy "..confusion.totalValid*100)
   end
   -- print confusion matrix
   --print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
   if epoch % opt.save_freq==0 then 
       paths.mkdir('compositionalMNISTclassifier')
       torch.save('checkpoints_compositionalMNISTclassifier/' .. opt.name .. 'vgg_' .. epoch .. '_net.t7', model )
    end  
   -- torch.save(filename, model)
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
    epoch, opt.niter, epoch_tm:time().real))
   -- next epoch
   epoch = epoch + 1
end

-- test function
function test(dataloader)
   -- local vars
   local time = sys.clock()
   -- test over given dataset
   print('<trainer> on testing Set:')
   for n, sample in dataloader:run() do
      -- disp progress
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,3,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      input:resize(sample.input:size()):copy(sample.input)
      label:resize(sample.target:size()):copy(sample.target)
      -- test samples
      local preds = model:forward(input)
      
      -- confusion:
      for i = 1,opt.batchSize do
         confusion:add(preds[i], label[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / dataloader:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   --print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()
end

----------------------------------------------------------------------
-- and train!
--
for i = 1, opt.niter do
   -- train/test
   train(trainLoader)
   test(valLoader)

   -- plot errors
   if opt.plot == 1 then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
end
