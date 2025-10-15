from matplotlib import pyplot as plt
import numpy as np 
uav_num = 1
user_num = 8 
users_path = r'/home/lancegan/Datas/Codes/Python/P2_New/results/datas/Users_7.txt'
f = open(users_path, 'r')
x0_user = []
y0_user = []
if f:
    for j in range(user_num):
        user_loc = f.readline()
        # print("user_loc", user_loc)
        user_loc = user_loc.split(' ')
        x0_user.append(float(user_loc[0]))
        # print("x_user",x_user)
        y0_user.append(float(user_loc[1]))
if __name__ == '__main__':
    
    method = 'DDPG'
    Radio_Map = 'A2G'
    
    if method =='Ours':
        np_ours_100 = np.load('results/trajectory/ours_100MB.npz')
        np_ours_200 = np.load('results/trajectory/ours_200MB.npz')
        np_ours_300 = np.load('results/trajectory/ours_300MB.npz')
    elif method =='PSO':
        np_ours_100 = np.load('results/trajectory/PSO_100MB.npz')
        np_ours_200 = np.load('results/trajectory/PSO_200MB.npz')
        np_ours_300 = np.load('results/trajectory/PSO_300MB.npz')
    elif method == 'ACO':
        np_ours_100 = np.load('results/trajectory/ACO_100MB.npz')
        np_ours_200 = np.load('results/trajectory/ACO_200MB.npz')
        np_ours_300 = np.load('results/trajectory/ACO_300MB.npz')
    elif method == 'GA':
        np_ours_100 = np.load('results/trajectory/GA_100MB.npz')
        np_ours_200 = np.load('results/trajectory/GA_200MB.npz')
        np_ours_300 = np.load('results/trajectory/GA_300MB.npz')
    elif method == 'DDPG':
        np_ours_100 = np.load('results/trajectory/DDPG_100MB.npz')
        np_ours_200 = np.load('results/trajectory/DDPG_200MB.npz')
        np_ours_300 = np.load('results/trajectory/DDPG_300MB.npz')
        
    x_ours_100 = np.transpose(np_ours_100['x0_uav'])*100
    y_ours_100 = np.transpose(np_ours_100['y0_uav'])*100
    T_ours_100 = np.transpose(np_ours_100['T'])
    
    x_ours_200 = np.transpose(np_ours_200['x0_uav'])*100
    y_ours_200 = np.transpose(np_ours_200['y0_uav'])*100
    T_ours_200 = np.transpose(np_ours_200['T'])
    
    x_ours_300 = np.transpose(np_ours_300['x0_uav'])*100
    y_ours_300 = np.transpose(np_ours_300['y0_uav'])*100
    T_ours_300 = np.transpose(np_ours_300['T'])
    
    
    #画Radio Map 
    fig_1 = plt.figure(30)
    if Radio_Map == 'A2G':
        npzfile = np.load('results/datas/Radio_datas_A2G.npz')
        OutageMapActual = npzfile['arr_0']
        OutageMapActual_SINR = npzfile['arr_1']
        X_vec = npzfile['arr_2']  # [0,1....100]标号
        Y_vec = npzfile['arr_3']
        plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, OutageMapActual_SINR)
        v = np.arange(-20, 36, 4)
        cbar = plt.colorbar(ticks=v)
        cbar.set_label('SNR', labelpad=20, rotation=270, fontsize=14)
    elif Radio_Map =='G2A':
        npzfile = np.load('results/datas/Radio_datas.npz')
        OutageMapActual = npzfile['arr_0']
        OutageMapActual_SINR = npzfile['arr_1']
        X_vec = npzfile['arr_2']  # [0,1....100]标号
        Y_vec = npzfile['arr_3']
        plt.contourf(np.array(X_vec) * 10, np.array(Y_vec) * 10, 1 - OutageMapActual)
        v = np.linspace(0, 1.0, 11, endpoint=True)
        cbar = plt.colorbar(ticks=v)
        cbar.set_label('coverage probability', labelpad=20, rotation=270, fontsize=14)

    
    for i in range(user_num): # 画用户位置
        if i!=0 and i!= user_num-1:
            if i == 1 :
                plt.scatter(x0_user[i]*1000, y0_user[i]*1000, c='black', marker='^',s=60,label='Inspection Point')
            else :
                plt.scatter(x0_user[i]*1000, y0_user[i]*1000, c='black', marker='^',s=60)
        else :
            if i == 0:
                plt.scatter(x0_user[i]*1000, y0_user[i]*1000, c='red', marker='o',s=60,label='Start') #画起始点
            else :
                plt.scatter(x0_user[i]*1000, y0_user[i]*1000, c='orange', marker='o',s=60,label='End') #画终点
    
    
    #绘制无人机轨迹
    plt.plot(x_ours_100[0][0:T_ours_100+1], y_ours_100[0][0:T_ours_100+1],c='black',label="I=100MB")
    plt.plot(x_ours_200[0][0:T_ours_200+1], y_ours_200[0][0:T_ours_200+1],c='magenta',label="I=200MB")
    plt.plot(x_ours_300[0][0:T_ours_300+1], y_ours_300[0][0:T_ours_300+1],c='blue',label="I=300MB")
    plt.legend(fontsize = 6)
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    if method == 'Ours':
        title_str = 'DRL-IGA Trajectory'
    elif method == 'PSO':
        title_str = 'DRL-PSO Trajectory'
    elif method == 'ACO':
        title_str = 'DRL-ACO Trajectory'
    elif method == 'GA':
        title_str = 'DRL-GA Trajectory'
    elif method == 'DDPG':
        title_str = 'DDPG-IGA Trajectory'
    else:
        title_str = f"{method} Trajectory"

    # plt.title(title_str + f" ({Radio_Map})", fontsize=16)
    # save_path = 'results/figs/'+str(method)+'_trajectory_'+str(Radio_Map)+'.pdf' 
    # plt.savefig(save_path, format='pdf', bbox_inches='tight')
    save_path = 'results/figs/'+str(method)+'_trajectory_'+str(Radio_Map)+'.jpg' 
    plt.savefig(save_path, format='jpg', bbox_inches='tight')
    plt.close()
    