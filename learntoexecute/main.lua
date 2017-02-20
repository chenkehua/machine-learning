--[[
  Copyright 2014 Google Inc. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
]]--
require "env"
include "data.lua"
include "utils/strategies.lua"
include "layers/MaskedLoss.lua"
include "layers/Embedding.lua"
require "nngraph"
function lstm(i, prev_c, prev_h)
  function new_input_sum()
    local i2h            = nn.Linear(params.rnn_size, params.rnn_size) --params.rnn_size input params.rnn_size output
    local h2h            = nn.Linear(params.rnn_size, params.rnn_size)
    return nn.CAddTable()({i2h(i), h2h(prev_h)}) --两个tensor相加
  end
  local in_gate          = nn.Sigmoid()(new_input_sum())
  local forget_gate      = nn.Sigmoid()(new_input_sum())
  local in_gate2         = nn.Tanh()(new_input_sum())
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_gate2})
  })
  local out_gate         = nn.Sigmoid()(new_input_sum())
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})--这个函数的输入为几个向量的table（在lua里所有列表、数组、矩阵都是table），输出这个table里各个向量component-wise的乘积
  return next_c, next_h
end

function create_network()
  local x                = nn.Identity()()  --Creates a module that returns whatever is input to it as output.This is useful when combined with the module ParallelTable in case you do not wish to do     anything to one of the input Tensors.这个函数建立一个输入模块，什么都不做，通常用在神经网络的输入层。用法如下：
  local y                = nn.Identity()()  --输出你输入的
  local prev_s           = nn.Identity()()
  local i                = {[0] = Embedding(symbolsManager.vocab_size,params.rnn_size)(x)}  --Embedding:__init(inputSize, outputSize)42*400
  local next_s           = {}
  local splitted         = {prev_s:split(2 * params.layers)}--params.layers=2
  for layer_idx = 1, params.layers do    --params.layers =2
    local prev_c         = splitted[2 * layer_idx - 1]
    local prev_h         = splitted[2 * layer_idx]
    local dropped        = nn.Dropout()(i[layer_idx - 1]) --Dropout是指在模型训练时随机让网络某些隐含层节点的权重不工作，不工作的那些节点可以暂时认为不是网络结构的一部分，但是它的权重得保留下来（只是暂时不更新而已） http://www.cnblogs.com/tornadomeet/p/3258122.html      
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, symbolsManager.vocab_size)--400 42   标志
  local pred             = nn.LogSoftMax()(h2y(i[params.layers]))  --LogSoftMax层，用来把上一层的响应归一化到0至1之间
  local err              = MaskedLoss()({pred, y})                 -- MaskedLoss:updateOutput(input)
  local module           = nn.gModule({x, y, prev_s},{err, nn.Identity()(next_s)})  --这里只是对两个输入进行了合并，然后进行输出，
  module:getParameters():uniform(-params.init_weight, params.init_weight)--init.weight=0.08,uniform() fill with random values between (a,b)-0.08 ,0.08
  --graph.dot(module.fg, 'MLP','/home/ckh/Documents/luajit/test/module')
  if params.gpuidx > 0 then--gpuidx=1
    return module:cuda()
  else
  -- graph.dot(module.fg,"MLP")
    return module
  end
  --graph.dot(module.fg, 'MLP','/home/ckh/Documents/luajit/test/module')
end

function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network()
  --graph.dot(core_network.fg, 'MLP','/home/ckh/Documents/luajit/test/core_network')
  paramx, paramdx = core_network:getParameters() --paramx是core_network里面所有可调参数的集合，paramdx是每个参数对loss的偏导数。需要注意的是这里的paramx和paramdx都相当于C++里面的“引用”，一旦你对他们进行了操作，模型里的参数也会跟着改变。
  model = {}
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do--seq_length=50      对model.s进行操作，变成{["ds"] = table[0],["s"] = table[51],["start_s"] = table[0]}
    model.s[j] = {}
    for d = 1, 2 * params.layers do--layers=2
      model.s[j][d] = torch.zeros(params.batch_size, params.rnn_size)  --100*400 y = torch.zeros(n) returns a one-dimensional Tensor of size n filled with zeros.
      if params.gpuidx > 0 then
        model.s[j][d] = model.s[j][d]:cuda()--convert a CPU model to GPU
      end
    end
  end
  for d = 1, 2 * params.layers do --对model.ds和model.start_s进行操作{["ds"] = sequence[4],["s"] = table[51],["start_s"] = sequence[4]}
    model.start_s[d] = torch.zeros(params.batch_size, params.rnn_size)
    model.ds[d] = torch.zeros(params.batch_size, params.rnn_size)
    if params.gpuidx > 0 then
      model.start_s[d] = model.start_s[d]:cuda()
      model.ds[d] = model.ds[d]:cuda()
    end
  end
  model.core_network = core_network
  -- mark
  model.rnns = cloneManyTimes(core_network, params.seq_length)--50创建给定网络的克隆。克隆与原始网络共享所有权重和gradWeights。累积渐变适当地累加渐变。
  --local rnnn=model.rnns
  --graph.dot(model.s.fg, 'MLP','/home/ckh/Documents/luajit/test/model.s')
  model.norm_dw = 0
  reset_ds()--把#model.ds全置为0
end

function reset_state(state)
  load_data(state)
  state.pos = 1
  state.acc = 0
  state.count = 0
  state.normal = 0
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

function reset_ds()
  for d = 1, #model.ds do--#model.ds
    model.ds[d]:zero()
  end
end

function fp(state, paramx_)     --fp算法么
  if paramx_ ~= paramx then paramx:copy(paramx_) end
  copy_table(model.s[0], model.start_s)  --把model.start_s复制到model.s[0]
  if state.pos + params.seq_length > state.data.x:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    tmp, model.s[i] = unpack(model.rnns[i]:forward({state.data.x[state.pos],   --估计是拆分的意思
                                                    state.data.y[state.pos + 1],
                                                    model.s[i - 1]}))
    if params.gpuidx > 0 then
      cutorch.synchronize()    --如果params.gpuidx 大于0，使得其同步发生
    end
    state.pos = state.pos + 1
    state.count = state.count + tmp[2]
    state.normal = state.normal + tmp[3]
  end
  state.acc = state.count / state.normal
  copy_table(model.start_s, model.s[params.seq_length])--把 model.s[params.seq_length]复制model.start_s
end

function bp(state)
  paramdx:zero()
  reset_ds()
  local tmp_val
  if params.gpuidx > 0 then tmp_val = torch.ones(1):cuda() else tmp_val = torch.ones(1) end
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local tmp = model.rnns[i]:backward({state.data.x[state.pos],
                                        state.data.y[state.pos + 1],
                                        model.s[i - 1]},
                                        { tmp_val, model.ds})[3]
    copy_table(model.ds, tmp)
    if params.gpuidx > 0 then
      cutorch.synchronize()
    end
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
end

function eval_training(paramx_)
  fp(state_train, paramx_)
  bp(state_train)
  return 0, paramdx
end

function run_test(state)
  reset_state(state)
  for i = 1, (state.data.x:size(1) - 1) / params.seq_length do
    fp(state, paramx)
  end
end

function show_predictions(state)
  reset_state(state)
  copy_table(model.s[0], model.start_s)  --仅仅是复制函数而已
  local input = {[1] = ""}
  local prediction = {[1] = ""}
  local sample_idx = 1
  local batch_idx = random(params.batch_size)--100
  for i = 1, state.data.x:size(1) - 1 do                        --rnn的forward算法
    local tmp = model.rnns[1]:forward({state.data.x[state.pos],
                                              state.data.y[state.pos + 1],
                                              model.s[0]})[2]
    if params.gpuidx > 0 then
      cutorch.synchronize()--为了运行挂钟时间，您应该在每个时间检查点之前调用cutorch.synchronize（）：
    end
    copy_table(model.s[0], tmp)
    local current_x = state.data.x[state.pos][batch_idx]
    input[sample_idx] = input[sample_idx] ..                              --每次循环input的table里面都会产生一个字母，直到结束
                        symbolsManager.idx2symbol[current_x]    --此处“..”为连接的作用
    local y = state.data.y[state.pos + 1][batch_idx]       --一旦input里面出现@，y就不等于0了  接着input中就有table【2】了
    if y ~= 0 then                                                                      --一旦prediction完了，y就为0了         一旦出现了.就完了
      local fnodes = model.rnns[1].forwardnodes
      local pred_vector = fnodes[#fnodes].data.mapindex[1].input[1][batch_idx]
      prediction[sample_idx] = prediction[sample_idx] ..
                               symbolsManager.idx2symbol[argmax(pred_vector)]
    end
    state.pos = state.pos + 1
    local last_x = state.data.x[state.pos - 1][batch_idx]
    if state.pos > 1 and symbolsManager.idx2symbol[last_x] == "." then --每个语句以.为结束
      if sample_idx >= 3 then                                            --要到大于3才能够跳出必须要有3个例子
        break
      end
      sample_idx = sample_idx + 1                                    --一直到大于3才能够跳出
      input[sample_idx] = ""                                               --重新来
      prediction[sample_idx] = ""
    end
  end
  io.write(string.format("Some exemplary predictions for the %s dataset\n",
                          state.name))
  for i = 1, #input do
    input[i] = input[i]:gsub("#", "\n\t\t     ")--替代
    input[i] = input[i]:gsub("@", "\n\tTarget:      ")
    io.write(string.format("\tInput:\t     %s", input[i]))
    io.write(string.format("\n\tPrediction:  %s\n", prediction[i]))
    io.write("\t-----------------------------\n")
  end
end

function main()
  
  local cmd = torch.CmdLine()--赋值的语句
  cmd:option('-gpuidx', 1, 'Index of GPU on which job should be executed. 0 for CPU')
  cmd:option('-target_length', 6, 'Length of the target expression.')
  cmd:option('-target_nesting', 3, 'Nesting of the target expression.')
  -- Available strategies: baseline, naive, mix, blend.
  cmd:option('-strategy', 'blend', 'Scheduling strategy.')
  cmd:text()
  local opt = cmd:parse(arg)

  params = {batch_size=100,
            seq_length=50,
            layers=2,
            rnn_size=400,
            init_weight=0.08,
            learningRate=0.5,
            max_grad_norm=5,
            target_length=opt.target_length,
            target_nesting=opt.target_nesting,
            target_accuracy=0.95,
            current_length=1,
            current_nesting=1,
            gpuidx = opt.gpuidx}

  init_gpu(opt.gpuidx)
  state_train = {hardness=_G[opt.strategy],      --function() file:///home/ckh/Documents/luajit/test/@/home/ckh/Documents/luajit/test/utils/strategies.lua   46
    len=math.max(10001, params.seq_length + 1),   --opt.strategy=blend    opt {["gpuidx"] = 1,["strategy"] = "blend",["target_length"] = 6,["target_nesting"] = 3}
    seed=1,
    kind=0,
    batch_size=params.batch_size,  --100
    name="Training" }
  state_val =   {hardness=current_hardness,  --function()  file:///home/ckh/Documents/luajit/test/@/home/ckh/Documents/luajit/test/utils/strategies.lua   55
    len=math.max(501, params.seq_length + 1),
    seed=1,
    kind=1,
    batch_size=params.batch_size,
    name="Validation" }

  state_test =  {hardness=target_hardness,   --function() file:///home/ckh/Documents/luajit/test/@/home/ckh/Documents/luajit/test/utils/strategies.lua  59
    len=math.max(501, params.seq_length + 1),
    seed=1,
    kind=2,
    batch_size=params.batch_size,
    name="Test"}
   print("Network parameters:")
   print(params)
  local states = {state_train, state_val, state_test }

  for _, state in pairs(states) do
    reset_state(state)
    assert(state.len % params.seq_length == 1)
  end
  setup()         --此处开始建立network
  local step = 0
  local epoch = 0
  local train_accs = {}
  local total_cases = 0
  local start_time = torch.tic()
  print("Starting training.")
  print(params.learningRate)
  while true do
    local epoch_size = floor(state_train.data.x:size(1) / params.seq_length) --函数返回不大于参数X的最大整数 state_train.data.x:size(1) =10001  params.seq_length=50 epoch_size=200
    step = step + 1
    if step % epoch_size == 0 then
      state_train.seed = state_train.seed + 1
      load_data(state_train)
    end                                                                                                                                     --optim.adam为bp+sdg算法
    optim.adam(eval_training, paramx, {learningRate=params.learningRate}, {})--function(paramx_)file:///home/ckh/Documents/luajit/test/@/home/ckh/Documents/luajit/test/main.lua 170 一阶梯度优化的算法随机目标函数
    total_cases = total_cases + params.seq_length * params.batch_size--50*100            evaluation_training:a function that takes a single input X , the point of a evaluation, and returns f(X) and df/dX >  x : the initial point
    epoch = ceil(step / epoch_size)      --lua中的一个函数,math.ceil(x)返回大于参数x的最小整数,即对浮点数向上取整  step要是没有办法大于epoch_size 200的话没戏
    if step % ceil(epoch_size / 2) == 10 then
      cps = floor(total_cases / torch.toc(start_time)) --函数返回不大于参数X的最大整数
      run_test(state_val)
      run_test(state_test)
      local accs = ""
      for _, state in pairs(states) do
        accs = string.format('%s, %s acc.=%.2f%%',
          accs, state.name, 100.0 * state.acc)
      end
      print('epoch=' .. epoch .. accs ..
        ', current length=' .. params.current_length ..
        ', current nesting=' .. params.current_nesting ..
        ', characters per sec.=' .. cps ..
        ', learning rate=' .. string.format("%.3f", params.learningRate))
      if (state_val.acc > params.target_accuracy) or
        (#train_accs >= 5 and
        train_accs[#train_accs - 4] > state_train.acc) then
        if not make_harder() then
          params.learningRate = params.learningRate * 0.8
        end
        if params.learningRate < 1e-3 then       --有可能是params.learningRate<其的时候全部推迟  learning rate=0.001  
          break
        end
        load_data(state_train)
        load_data(state_val)
        train_accs = {}
      end
      train_accs[#train_accs + 1] = state_train.acc
      total_cases = 0
      start_time = torch.tic()
      show_predictions(state_train)
      show_predictions(state_val)
      show_predictions(state_test)
    end
    if step % 33 == 0 then
      collectgarbage()  --垃圾回收机制
    end
  end
  print("Training is over.")
  print(params.learningRate)
end

main()
