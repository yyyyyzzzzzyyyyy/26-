import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
import time
import tracemalloc
tracemalloc.start()  # 启动内存监控
# 固定参数
BETA_LIST = [0.7, 0.8, 0.85, 0.9, 0.95]  # 对比用系数
BETA = 0.85          # 阻尼因子
EPS = 1e-8           # 迭代收敛阈值，小于此值停止迭代
MAX_ITER = 1000      # 最大迭代轮次上限，防止死循环
TOP_K = 100          # 输出排名前100节点
INPUT_FILE = "Data.txt"   # 输入数据文件
OUTPUT_FILE = "Res.txt"   # 结果输出文件
BLOCK_SIZE = 1000    # 分块矩阵大小，内存优化核心参数
# 解决matplotlib中文显示问题
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
#构建标准稀疏转移矩阵
def load_matrix(file_path):
    edge_list=[]       # 存储所有原始有向边
    out_degree=dict()  # 统计每个节点的出度
    allnodes=set()    # 统计数据集内所有出现过的节点
    # 逐行读取数据
    with open(file_path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            u, v = map(int, line.split())
            edge_list.append((u, v))
            allnodes.add(u)
            allnodes.add(v)
            out_degree[u] = out_degree.get(u, 0) + 1
    # 节点ID映射：把原始大数字ID转为连续0开始的索引
    nodes=list(allnodes)
    idx={n: i for i,n in enumerate(nodes)}
    N=len(nodes)
    print(f"数据集总节点数：{N}")
    print(f"数据集总边数：{len(edge_list)}")
    # M[i][j]=节点j到节点i的跳转权重=1/节点j的出度
    # 只有存在边j→i时才有值，其余全部0元素不存储
    row=[]
    col=[]
    data=[]
    for u, v in edge_list:
        j=idx[u]  # 源节点u→矩阵列
        i=idx[v]  # 目标节点v→矩阵行
        weight=1.0/out_degree[u]
        row.append(i)
        col.append(j)
        data.append(weight)
    sparse_M=csr_matrix((data,(row, col)),shape=(N, N))
    return sparse_M,nodes,N,out_degree,idx
#死端节点预处理
def get_dead_end(out_degree,idx):
    dead_idx=[]
    for node in idx.keys():
        if node not in out_degree:
            dead_idx.append(idx[node])
    print(f"数据集中死端节点总数 = {len(dead_idx)}")
    return dead_idx

#标准分块矩阵迭代PageRank计算
def pagerank(sparse_M, N,dead_idx,beta,eps,max_iter, block_size):
    # 初始化：所有节点平分初始总分1
    pr_old=np.ones(N)/N
    # 阻尼固定常数项：用户随机跳转项，解决蜘蛛陷阱
    random_item=(1-beta)/N
    iter_count=max_iter  # 先默认是最大迭代次数
    for iter_round in range(max_iter):
        pr_new=np.zeros(N)
        # 汇总所有死端节点的全部分数
        dead_total_score=np.sum(pr_old[dead_idx])
        # 全部死端分数平均分给全图每一个节点
        dead_compensate=dead_total_score/N
        # 对大矩阵分块，逐块计算，实现分块内存优化
        for start in range(0,N,block_size):
            end=min(start+block_size,N)
            # 仅加载当前块子矩阵进内存
            block_M=sparse_M[start:end,:]
            # 分块矩阵乘法计算
            pr_new[start:end]=beta*block_M.dot(pr_old)
        pr_new+=random_item+beta*dead_compensate
        #收敛性判断
        delta=np.sum(np.abs(pr_new - pr_old))
        pr_old=pr_new.copy()
        if delta<eps:
            print(f"迭代收敛,总共迭代轮次：{iter_round+1} 轮")
            iter_count=iter_round+1  # 更新实际迭代次数
            break
    else:
        print(f"达到最大迭代次数{max_iter}轮，终止迭代")
    return pr_old,iter_count

# 保存β=0.85的结果
def save_top100(pr_score, node_list):
    node_score=sorted([(node_list[i],pr_score[i]) for i in range(len(node_list))], key=lambda x:x[1], reverse=True)
    with open(OUTPUT_FILE,'w',encoding='utf-8') as f:
        for node, score in node_score[:TOP_K]:
            f.write(f"{node} {score}\n")
    print(f"β=0.85 结果已保存至 {OUTPUT_FILE}")
# 绘制对比图表
def plot_compare(betas, iters, times):
    fig,(ax1,ax2)=plt.subplots(1, 2, figsize=(12, 4))
    # 子图1：阻尼系数vs迭代次数
    ax1.plot(betas, iters, 'o-', color='#2E86AB', linewidth=2, markersize=6)
    ax1.set_xlabel("阻尼系数 β", fontsize=12)
    ax1.set_ylabel("迭代次数", fontsize=12)
    ax1.set_title("β - 迭代次数 对比", fontsize=14)
    ax1.grid(alpha=0.3)
    # 子图2：阻尼系数vs运行时间
    ax2.plot(betas, times, 'o-', color='#A23B72', linewidth=2, markersize=6)
    ax2.set_xlabel("阻尼系数 β", fontsize=12)
    ax2.set_ylabel("运行时间 (秒)", fontsize=12)
    ax2.set_title("β - 运行时间 对比", fontsize=14)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("β系数对比图.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("对比图表已保存为：β系数对比图.png")

def main():
    start_all=time.time()
    M,node_list,N,out_degree,nodeidx=load_matrix(INPUT_FILE)
    dead_idx=get_dead_end(out_degree,nodeidx)
    # 存储对比数据
    iter_list,time_list=[],[]
    final_pr_main=None
    for beta in BETA_LIST:
        t0=time.time()
        pr,it=pagerank(M,N,dead_idx,beta,EPS,MAX_ITER,BLOCK_SIZE)
        t1=time.time()
        cost=round(t1-t0,2)
        iter_list.append(it)
        time_list.append(cost)
        # 仅保存β=0.85的结果
        if beta==BETA:
            final_pr_main=pr
        print(f"β={beta}|迭代：{it}次|耗时：{cost}s")
    # 保存结果
    save_top100(final_pr_main,node_list)
    # 绘图
    plot_compare(BETA_LIST,iter_list,time_list)
    print(f"\n总耗时：{round(time.time()-start_all, 2)}s")
    # 输出最大运行内存
    peak_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    print(f"程序运行【峰值内存占用】：{peak_memory:.2f} MB")
    tracemalloc.stop()
if __name__ == "__main__":
    main()
