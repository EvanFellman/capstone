#python naive_ranker_test.py --hotpot ~/data/hotpot_train_v1.1.json --train 3 --test 1000  --epochs 1 --tune

with open("make_table.bash", "w") as f:
    for train_len in [0,1,2,3,4,5,10,20,50,100,1000]:
        for epochs in [1,3,5]:
            #write the following command on a new line
            #python naive_ranker_test.py --hotpot ~/data/hotpot_train_v1.1.json --train train_len --test 1000  --epochs epochs --tune
            if train_len == 0:
                f.write(f"python naive_ranker_test.py --hotpot ~/data/hotpot_train_v1.1.json --train {train_len} --test 1000  --epochs {epochs}\n")
            else:
                f.write(f"python naive_ranker_test.py --hotpot ~/data/hotpot_train_v1.1.json --train {train_len} --test 1000  --epochs {epochs} --tune\n")
            
