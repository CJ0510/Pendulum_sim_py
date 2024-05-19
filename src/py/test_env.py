import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
import random

class Pendulum():

    def __init__(self):
        self.seed = random.randint(0, 100) #启动选项
        self.eng = matlab.engine.start_matlab(str(self.seed)) #启动matlab环境
        # 动作空间维度
        self.action_space_n = 1
        self.observation_space_n = 3
        # 动作的范围
        self.action_space_low = -2.0
        self.action_space_high = 2.0
        self.action_space = np.array([self.action_space_low , self.action_space_high])
        # 总时间Tf，采样时间Ts
        self.Ts = 0.05
        self.Tf = 20
        self.max_steps = int(self.Tf/self.Ts)

    ''' 初始化状态重置'''
    def reset(self):
        # 采样时间
        self.eng.workspace['Ts'] = self.Ts
        # 总时间
        self.eng.workspace['Tf'] = self.Tf
        self.eng.workspace['model_name'] = 'rlSimplePendulumModelmy'
        #模型文件加载
        self.eng.eval('load_system(model_name)',nargout=0)
        #初始化动作
        self.eng.workspace['action'] = 0
        #设置初始化动作
        self.eng.eval("set_param('rlSimplePendulumModelmy/Gain', 'Gain', num2str(action))", nargout=0)
        # 设置仿真时间 Tf
        self.eng.eval("set_param(model_name, 'StopTime', num2str(Tf))", nargout=0)
        # 设置定步长求解，每次采样时间Ts
        self.eng.eval("set_param(model_name, 'SolverType', 'Fixed-step', 'FixedStep', num2str(Ts))", nargout=0)
        # 开始仿真
        self.eng.eval("set_param(model_name, 'SimulationCommand', 'start')", nargout=0)
        # 停止仿真
        self.eng.eval("set_param(model_name, 'SimulationCommand', 'pause')", nargout=0)
        # 取所有状态
        state = self.eng.eval('out.observation')
        #取最后一个状态，格式转换
        state = np.array(state[-1]).astype(float)
        return state

    ''' 步进执行 '''
    def step(self,action):
        # 连续系统，步进执行，返回s_、r、isdone等参数情况
        # 通过在中途改变模块的值来间接改变输入值
        self.eng.workspace['action'] = float(action) # 将action 写入workspace中
        self.eng.eval("set_param('rlSimplePendulumModelmy/Gain', 'Gain', num2str(action))", nargout=0)
        self.eng.eval("set_param(model_name, 'SimulationCommand', 'step')", nargout=0)
        next_state = self.eng.eval('out.observation')
        reward= self.eng.eval('out.reward')
        done =self.eng.eval('out.isdone')
        next_state = np.array(next_state[-1]).astype(float)
        reward = np.array(reward[-1]).astype(float)
        done = np.array(done[-1]).astype(bool)
        return next_state, reward,done

    ''' 停止仿真 '''
    def stop(self):
        self.eng.eval("set_param(model_name, 'SimulationCommand', 'stop')", nargout=0)
        # 清屏，清除工作空间的变量，方便下一次重新启动
        self.eng.eval("clc", nargout=0)
        self.eng.eval("clear", nargout=0)

    ''' 随机选择动作输出 '''
    def action_sample(self):
        action = np.random.uniform(self.action_space_low,self.action_space_high,self.action_space_n)
        return action

    ''' 退出Matlab-simulink释放进程 '''
    def exit(self):
        self.eng.quit()
        self.eng.exit()


if __name__ == '__main__':
    env = Pendulum()
    for i in range(1):
        s=env.reset()
        rs=np.zeros((3,1))
        for j in range(env.max_steps):
            a = env.action_sample()
            s, r, d = env.step(a)
            print("s:{}\nr:{}\nd{}".format(s,r,d))
            print("s.shape:{}\nr.shape:{}\nd.shape{}".format(s.shape,r.shape,d.shape))
            rs+=r
            if j==(env.max_steps-1):
                env.stop()
                print((i+1),'reward',rs)
                break
    env.exit()





