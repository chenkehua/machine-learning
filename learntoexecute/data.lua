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
include "utils/operations.lua"
include "utils/stack.lua"
include "utils/symbolsManager.lua"
include "utils/variablesManager.lua"
include "utils/utils.lua"

local stack = Stack()
variablesManager = VariablesManager()
symbolsManager = SymbolsManager()

function to_data(code, var, output)
  local x = {}
  local y = {}
  local output = string.format("%d.", output)
  local input = ""
  for i = 1, #code do
    input = string.format("%s%s#", input, code[i])
  end
  input = string.format("%sprint(%s)@", input, var)
  for j = 1, #input do
    table.insert(y, 0)
    table.insert(x, symbolsManager:get_symbol_idx(input:byte(j)))
  end
  for j = 1, #output do
    table.insert(x, symbolsManager:get_symbol_idx(output:byte(j)))
    table.insert(y, symbolsManager:get_symbol_idx(output:byte(j)))
  end
  local orig = string.format("%s%s", input, output)
  return {x, y, orig}
end

function compose(hardness)
  stack:clean()--clear stack站清空
  variablesManager:clean()--  self.vars = {}  self.last_var_idx = 0
  local funcs = {}
  local names = {}
  for i, v in pairs(_G) do
    if (string.find(i, "_opr") ~= nil) then--存在一个{"pair_opr","small_loop_opr","ifstat_opr","equality_opr","smallmul_opr","vars_opr"}中的一个时即使有opr
      funcs[#funcs + 1] = v
      names[#names + 1] = i
    end
  end                  --{"pair_opr","small_loop_opr","ifstat_opr","equality_opr","smallmul_opr","vars_opr"}  names
  local code = {}
  local hard, nest = hardness() --params.current_length  params.current_nesting   1,1
  for h = 1, nest do
    local f_idx = random(#funcs)
    local f = funcs[f_idx]
    local code_tmp, var_tmp, output_tmp = f(hardness)  --function(hardness) 此处想要生成函数
    for i = 1, #code_tmp do                                            --"(1 if 4>3 else 10)" var_tmp 完全随机生成的
      code[#code + 1] = code_tmp[i]
    end
    stack:push({var_tmp, output_tmp})            --把var_tmp放到了stack中 "(1 if 4>3 else 10)"输入 ，1为输出  
  end
  local var, output = unpack(stack:pop())
  return code, var, output -- input, target, orig 对应上一层的这些
end

function get_operand(hardness)
  if stack:is_empty() then
    local eval = random(math.pow(10, hardness()))
    local expr = string.format("%d", eval)
    return expr, eval
  else
    return unpack(stack:pop())
  end
end

function get_operands(hardness, nr)
  local ret = {}
  local perm = torch.randperm(nr)
  for i = 1, nr do
    local expr, eval = get_operand(hardness)
    ret[perm[i]] = {expr=expr, eval=eval}
  end
  return unpack(ret)--初始生成4,3,1,10
end

function get_data(state)
  make_deterministic(state.seed)--utils
  local len = state.len       --len
  local batch_size = state.batch_size  --state.batch_size
  if state.data == nil then
    state.data = {}
    state.data.x = torch.ones(len, batch_size)   --生成10001*100的矩阵  里面全是1
    state.data.y = torch.zeros(len, batch_size)  --生成10001*100的矩阵  里面全是0
  end
  local x = state.data.x    --  里面全是1
  local y = state.data.y   --   里面全是0
  local count = 0
  io.write(string.format("Few exemplary newly generated " ..
                         "samples from the %s dataset:\n", state.name))
  local idx = 1
  local batch_idx = 1
  local i = 0
  while true do
    data = to_data(compose(state.hardness))--生成函数～{sequence[28],sequence[28],"print((3 if 1>10 else 6))@6."}此处生成data如此类的东西  code, var, output
    input, target, orig = unpack(data)--input生成{1,2,3,4,5,6,6,25,27,3,28,27,20,29,20,15,27,26,30,31,26,27,24,8,8,9,24,10}东西  orig生成"print((3 if 1>10 else 6))@6."东西 target生成{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,10}类似东西
    if str_hash(orig) % 3 == state.kind then    --这里的函数很奇怪～为什么要满足这个
      count = count + #orig
      if idx + #input > x:size(1) then
        idx = 1
        batch_idx = batch_idx + 1
        if batch_idx > batch_size then
          break;
        end
      end
    --data   {sequence[24],sequence[24],"print((51713-38))@51675."}
    --input  {1,2,3,4,5,6,6,16,20,13,20,25,19,25,22,8,8,9,16,20,24,13,16,10}  24
    --target  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,20,24,13,16,10}                  24
    --orig    "print((51713-38))@51675."
      for j = 1, #input do          --对input这组数据进行分析{23,12,20,18,28,32,2,27,33,27,3,4,27,2,34,4,35,26,6,20,8,36,23,19,12,16,18,1,2,3,4,5,6,23,8,9,19,7,10}
        x[idx][batch_idx] = input[j]
        y[idx][batch_idx] = target[j]  --{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19,7,10}target
        idx = idx + 1
        -- Added to take care of smaller state lens
        if idx > x:size(1) then 
          idx = 1 
          batch_idx = batch_idx + 1
          if batch_idx > batch_size then 
            break
          end
        end
      end
      
      if batch_idx > batch_size then 
            break
      end 
      
      if (i <= 2) then
        io.write("\tInput:\t")
        local orig = string.format("     %s", orig)
        orig = orig:gsub("#", "\n\t\t     ")--grup用于替代掉#
        orig = orig:gsub("@", "\n\tTarget:      ")--用于替代掉@
        io.write(orig)
        io.write("\n")
        io.write("\t-----------------------------\n")
        i = i + 1
      end
    end
  end
  io.write("\n")
end

function load_data(state)
  if state.currently_loaded_seed == state.seed then--state.currently_loaded_seed == state.seed   false
    return
  else
    state.currently_loaded_seed = state.seed
    get_data(state)
  end
end

function hardness_fun()
  return 8, 4
end

if script_path() == "data.lua" then
  make_deterministic(1)
  print("Data verification")
  for k = 1, 1000 do
    code, var, output = compose(hardness_fun)
    output = string.format("%d", output)
    print("\n__________________\n")
    local input = ""
    for i = 1, #code do
      input = string.format("%s%s\n", input, code[i])
    end
    input = string.format("%sprint(%s)", input, var)
    print(string.format("Input: \n%s\n", input))
    print(string.format("Target: %s", output))
    lines = os.capture(string.format("python2.7 -c '%s'", input))
    print(lines)
    lines = string.sub(lines, 1, string.len(lines) - 1)
    if lines ~= output then
      print(string.format("\nERROR!\noutput from python: '%s', " ..
                          "doesn't match target output: '%s'", lines, output))
      exit(-1)
    end
  end
  print("\n__________________\n")
  print("Successfully verified coherence of generated a " .. 
        "targets with python interpreter.")
end
