

for p in -1 -0.7 -0.5 -0.3
do
    mkdir target_poly_$p
    cd target_poly_$p
    python ../main_simsiam.py \
          -a resnet50 \
          --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
          -b 512 \
          --fix-pred-lr \
          --pred-type poly \
          --sigma $p \
          --pred-location target \
          /home/zjzhuo/Project/datasets/imagenet

    python ../main_lincls.py \
          -a resnet50 \
          --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
          --pretrained checkpoint_0099.pth.tar \
          --lars \
          /home/zjzhuo/Project/datasets/imagenet

    cd ..

done



for type in poly log log_1
do
      mkdir online_$type
      cd online_$type
      python ../main_simsiam.py \
                  -a resnet50 \
                  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
                  -b 512 \
                  --fix-pred-lr \
                  --pred-type $type \
                  --sigma 2 \
                  --pred-location online \
                  /home/zjzhuo/Project/datasets/imagenet

      python ../main_lincls.py \
                  -a resnet50 \
                  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
                  --pretrained checkpoint_0099.pth.tar \
                  --lars \
                  /home/zjzhuo/Project/datasets/imagenet

      cd ..
done


