import os

f = open('vallist_ins.txt', 'r')
lines = f.readlines()
f.close()
for cnt,line in enumerate(lines):
    f = open('temp'+str(cnt)+'.txt','w')
    f.write(line)
    f.close()
    cmd = "python test_VIP_instance_FAST.py --cropSize 560 --gpu-id 0,1 --topk_vis 20 --resume checkpoint_14.pth.tar --save_path VIP_ins --evaluate --videoLen 2 --batch_idx {:n}".format(cnt)
    print(cmd)
    os.system(cmd)
