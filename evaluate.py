from graspnetAPI import GraspGroup, GraspNetEval
import numpy as np

ge = GraspNetEval(root="/hpcfiles/users/guihaiyuan/data/Benchmark/graspnet", camera="realsense", split='test')
# ge = GraspNetEval(root="/hpcfiles/users/guihaiyuan/data/Benchmark/graspnet", camera="kinect", split='test')

# res, ap = ge.eval_seen("logs/dump_sbg_kn_2_wo_CD", proc=10)
# print("seen")
# print("AP",np.mean(res))
# print(res.shape)
# np.save("logs/dump_sbg_kn_2_wo_CD/seen.npy",res)
# res = res.transpose(3,0,1,2).reshape(6,-1)
# res = np.mean(res,axis=1)
# print("AP0.4",res[1])
# print("AP0.8",res[3])

# # save_dir = os.path.join(cfgs.dump_dir, 'ap_{}.npy'.format(cfgs.camera))
# # np.save(save_dir, res)
# # res, ap = ge.eval_similar("logs/dump_sbg_kn_2", proc=32)
# # print("similar")
# # print("AP",np.mean(res))
# # np.save("logs/dump_sbg_kn_2/similar.npy",res)
# # res = res.transpose(3,0,1,2).reshape(6,-1)
# # res = np.mean(res,axis=1)
# # print("AP0.4",res[1])
# # print("AP0.8",res[3])

res, ap = ge.eval_novel("logs/dump_sbg_rs_4_obs_NcM", proc=20)
print("novel")
print("AP",np.mean(res))
np.save("logs/dump_sbg_rs_4_obs_NcM/novel.npy",res)
res = res.transpose(3,0,1,2).reshape(6,-1)
res = np.mean(res,axis=1)
print("AP0.4",res[1])
print("AP0.8",res[3])