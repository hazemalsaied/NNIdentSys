INFO: activate-binutils_linux-64.sh made the following environmental changes:
+ADDR2LINE=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-addr2line
+AR=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-ar
+AS=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-as
+CXXFILT=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-c++filt
+ELFEDIT=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-elfedit
+GPROF=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-gprof
+HOST=x86_64-conda_cos6-linux-gnu
+LD_GOLD=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-ld.gold
+LD=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-ld
+NM=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-nm
+OBJCOPY=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-objcopy
+OBJDUMP=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-objdump
+RANLIB=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-ranlib
+READELF=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-readelf
+SIZE=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-size
+STRINGS=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-strings
+STRIP=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-strip
INFO: activate-gcc_linux-64.sh made the following environmental changes:
+CC=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-cc
+CFLAGS=-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -pipe
+_CONDA_PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata_x86_64_conda_cos6_linux_gnu
+CPPFLAGS=-DNDEBUG -D_FORTIFY_SOURCE=2 -O2
+CPP=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-cpp
+DEBUG_CFLAGS=-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-all -fno-plt -Og -g -Wall -Wextra -fvar-tracking-assignments -pipe
+DEBUG_CPPFLAGS=-D_DEBUG -D_FORTIFY_SOURCE=2 -Og
+GCC_AR=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-gcc-ar
+GCC=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-gcc
+GCC_NM=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-gcc-nm
+GCC_RANLIB=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-gcc-ranlib
+LDFLAGS=-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now
INFO: activate-gxx_linux-64.sh made the following environmental changes:
+CXXFLAGS=-fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -pipe
+CXX=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-c++
+DEBUG_CXXFLAGS=-fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-all -fno-plt -Og -g -Wall -Wextra -fvar-tracking-assignments -pipe
+GXX=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-g++
	TransitionClassifier(
  (p_embeddings): Embedding(21, 40)
  (w_embeddings): Embedding(4640, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
	Language : FR
==================================================================================================
	Training (Important) : 11506, Test : 2541
	MWEs in tain : 7100, occurrences : 25744
	Impotant words in tain : 4638
	MWE length mean : 2.66
	Seen MWEs : 3130 (77 %)
	New MWEs : 909 (22 %)
==================================================================================================
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/423.kiperwasser.p
Number of used sentences in train = 10355
Total loss for epoch 0: 75612.499773
validation loss after epoch 0 : 7466.248636
	Epoch 1....
validAcc: 0.165
Epoch has taken 0:12:50.913462
Number of used sentences in train = 10355
Total loss for epoch 1: 56981.718638
validation loss after epoch 1 : 4147.476858
validAcc: 0.734
	Epoch 2....
Epoch has taken 0:13:19.361871
Number of used sentences in train = 10355
Total loss for epoch 2: 50740.020171
validation loss after epoch 2 : 4784.132808
	Epoch 3....
validAcc: 0.167
Epoch has taken 0:13:00.846728
Number of used sentences in train = 10355
Total loss for epoch 3: 47123.800534
validation loss after epoch 3 : 3437.552835
	Epoch 4....
validAcc: 0.27
Epoch has taken 0:14:17.875058
Number of used sentences in train = 10355
Total loss for epoch 4: 44465.356952
validation loss after epoch 4 : 3658.110884
	Epoch 5....
validAcc: 0.666
Epoch has taken 0:14:08.593240
Number of used sentences in train = 10355
Total loss for epoch 5: 42545.090828
validation loss after epoch 5 : 4218.332072
validAcc: 0.758
	Epoch 6....
Epoch has taken 0:13:44.682648
Number of used sentences in train = 10355
Total loss for epoch 6: 41029.791085
validation loss after epoch 6 : 4250.580559
	Epoch 7....
validAcc: 0.17
Epoch has taken 0:12:58.346474
Number of used sentences in train = 10355
Total loss for epoch 7: 39778.189002
validation loss after epoch 7 : 2960.393329
validAcc: 0.784
	Epoch 8....
Epoch has taken 0:12:52.052731
Number of used sentences in train = 10355
Total loss for epoch 8: 38802.744517
validation loss after epoch 8 : 4304.120396
	Epoch 9....
validAcc: 0.309
Epoch has taken 0:13:43.672899
Number of used sentences in train = 10355
Total loss for epoch 9: 38019.000039
validation loss after epoch 9 : 3205.412986
	Epoch 10....
validAcc: 0.166
Epoch has taken 0:14:06.161132
Number of used sentences in train = 10355
Total loss for epoch 10: 37310.175097
validation loss after epoch 10 : 2802.211683
	Epoch 11....
validAcc: 0.392
Epoch has taken 0:14:06.149540
Number of used sentences in train = 10355
Total loss for epoch 11: 36662.620714
validation loss after epoch 11 : 3172.941589
	TransitionClassifier(
  (p_embeddings): Embedding(21, 40)
  (w_embeddings): Embedding(4640, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
validAcc: 0.178
Epoch has taken 0:13:56.582835
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/0.kiperwasser.p
Number of used sentences in train = 1151
Total loss for epoch 0: 6104.106504
	Epoch 1....
Epoch has taken 0:01:26.711243
Number of used sentences in train = 1151
Total loss for epoch 1: 3381.623945
	Epoch 2....
Epoch has taken 0:01:26.723282
Number of used sentences in train = 1151
Total loss for epoch 2: 2458.656014
	Epoch 3....
Epoch has taken 0:01:26.741183
Number of used sentences in train = 1151
Total loss for epoch 3: 1888.228694
	Epoch 4....
Epoch has taken 0:01:26.724977
Number of used sentences in train = 1151
Total loss for epoch 4: 1578.242154
	Epoch 5....
Epoch has taken 0:01:26.761275
Number of used sentences in train = 1151
Total loss for epoch 5: 1348.381443
	Epoch 6....
Epoch has taken 0:01:26.701746
Number of used sentences in train = 1151
Total loss for epoch 6: 1219.698396
	Epoch 7....
Epoch has taken 0:01:26.680565
Number of used sentences in train = 1151
Total loss for epoch 7: 1114.669877
	Epoch 8....
Epoch has taken 0:01:26.702563
Number of used sentences in train = 1151
Total loss for epoch 8: 1025.294974
	Epoch 9....
Epoch has taken 0:01:26.740687
Number of used sentences in train = 1151
Total loss for epoch 9: 969.791904
	Epoch 10....
Epoch has taken 0:01:27.118325
Number of used sentences in train = 1151
Total loss for epoch 10: 921.767751
	Epoch 11....
Epoch has taken 0:01:26.861328
Number of used sentences in train = 1151
Total loss for epoch 11: 877.250761
Epoch has taken 0:01:20.497213

==================================================================================================
	Training time : 3:00:29.501859
==================================================================================================
	Identification : 0.021

*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|
	Language : FR
==================================================================================================
	Training (Important) : 10585, Test : 2541
	MWEs in tain : 6789, occurrences : 23628
	Impotant words in tain : 4468
	MWE length mean : 2.66
	Seen MWEs : 3102 (76 %)
	New MWEs : 937 (23 %)
==================================================================================================
	TransitionClassifier(
  (p_embeddings): Embedding(21, 40)
  (w_embeddings): Embedding(4470, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/421.kiperwasser.p
Number of used sentences in train = 9526
Total loss for epoch 0: 67852.236797
validation loss after epoch 0 : 6470.899522
	Epoch 1....
validAcc: 0.723
Epoch has taken 0:12:30.410422
Number of used sentences in train = 9526
Total loss for epoch 1: 51541.374247
validation loss after epoch 1 : 4719.376462
validAcc: 0.734
	Epoch 2....
Epoch has taken 0:12:54.870167
Number of used sentences in train = 9526
Total loss for epoch 2: 45861.946553
validation loss after epoch 2 : 4641.146544
	Epoch 3....
validAcc: 0.715
Epoch has taken 0:12:09.775563
Number of used sentences in train = 9526
Total loss for epoch 3: 42411.443119
validation loss after epoch 3 : 4271.861260
validAcc: 0.744
	Epoch 4....
Epoch has taken 0:11:37.401727
Number of used sentences in train = 9526
Total loss for epoch 4: 39849.154714
validation loss after epoch 4 : 4182.676128
	Epoch 5....
validAcc: 0.205
Epoch has taken 0:11:33.752544
Number of used sentences in train = 9526
Total loss for epoch 5: 37958.095782
validation loss after epoch 5 : 2711.252422
	Epoch 6....
validAcc: 0.301
Epoch has taken 0:11:36.371381
Number of used sentences in train = 9526
Total loss for epoch 6: 36574.538812
validation loss after epoch 6 : 3162.104062
	Epoch 7....
validAcc: 0.114
Epoch has taken 0:11:33.033632
Number of used sentences in train = 9526
Total loss for epoch 7: 35476.586016
validation loss after epoch 7 : 2320.687070
	Epoch 8....
validAcc: 0.199
Epoch has taken 0:11:42.887010
Number of used sentences in train = 9526
Total loss for epoch 8: 34637.314501
validation loss after epoch 8 : 2381.251692
	Epoch 9....
validAcc: 0.171
Epoch has taken 0:11:39.907036
Number of used sentences in train = 9526
Total loss for epoch 9: 33932.427652
validation loss after epoch 9 : 2415.976303
	Epoch 10....
validAcc: 0.732
Epoch has taken 0:11:38.808631
Number of used sentences in train = 9526
Total loss for epoch 10: 33339.909861
validation loss after epoch 10 : 3667.853910
	Epoch 11....
validAcc: 0.744
Epoch has taken 0:11:34.799169
Number of used sentences in train = 9526
Total loss for epoch 11: 32881.350015
validation loss after epoch 11 : 3436.264623
	TransitionClassifier(
  (p_embeddings): Embedding(21, 40)
  (w_embeddings): Embedding(4470, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
validAcc: 0.177
Epoch has taken 0:11:34.198162
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/0.kiperwasser.p
Number of used sentences in train = 1059
Total loss for epoch 0: 6240.435213
	Epoch 1....
Epoch has taken 0:01:13.262864
Number of used sentences in train = 1059
Total loss for epoch 1: 2366.791428
	Epoch 2....
Epoch has taken 0:01:13.330439
Number of used sentences in train = 1059
Total loss for epoch 2: 1573.830946
	Epoch 3....
Epoch has taken 0:01:12.121333
Number of used sentences in train = 1059
Total loss for epoch 3: 1260.737908
	Epoch 4....
Epoch has taken 0:01:13.396450
Number of used sentences in train = 1059
Total loss for epoch 4: 1049.367807
	Epoch 5....
Epoch has taken 0:01:11.256785
Number of used sentences in train = 1059
Total loss for epoch 5: 949.910553
	Epoch 6....
Epoch has taken 0:01:11.132330
Number of used sentences in train = 1059
Total loss for epoch 6: 871.842876
	Epoch 7....
Epoch has taken 0:01:11.117425
Number of used sentences in train = 1059
Total loss for epoch 7: 833.885942
	Epoch 8....
Epoch has taken 0:01:11.148557
Number of used sentences in train = 1059
Total loss for epoch 8: 804.389588
	Epoch 9....
Epoch has taken 0:01:11.148248
Number of used sentences in train = 1059
Total loss for epoch 9: 775.233547
	Epoch 10....
Epoch has taken 0:01:11.030850
Number of used sentences in train = 1059
Total loss for epoch 10: 748.609519
	Epoch 11....
Epoch has taken 0:01:11.066824
Number of used sentences in train = 1059
Total loss for epoch 11: 734.084346
Epoch has taken 0:01:11.081938

==================================================================================================
	Training time : 2:36:29.267011
==================================================================================================
	Identification : 0.019

*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|
	Language : FR
==================================================================================================
	Training (Important) : 10585, Test : 1235
	MWEs in tain : 6789, occurrences : 23628
	Impotant words in tain : 4468
	MWE length mean : 2.66
	Seen MWEs : 1753 (82 %)
	New MWEs : 363 (17 %)
==================================================================================================
	TransitionClassifier(
  (p_embeddings): Embedding(21, 40)
  (w_embeddings): Embedding(4470, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/253.kiperwasser.p
Number of used sentences in train = 9526
Total loss for epoch 0: 71402.168127
validation loss after epoch 0 : 6741.511564
	Epoch 1....
validAcc: 0.663
Epoch has taken 0:11:29.192174
Number of used sentences in train = 9526
Total loss for epoch 1: 53384.354514
validation loss after epoch 1 : 5527.585546
validAcc: 0.714
	Epoch 2....
Epoch has taken 0:11:34.223551
Number of used sentences in train = 9526
Total loss for epoch 2: 46932.138444
validation loss after epoch 2 : 5005.415734
	Epoch 3....
validAcc: 0.169
Epoch has taken 0:11:31.223335
Number of used sentences in train = 9526
Total loss for epoch 3: 42964.613917
validation loss after epoch 3 : 2753.372426
	Epoch 4....
validAcc: 0.434
Epoch has taken 0:11:39.729867
Number of used sentences in train = 9526
Total loss for epoch 4: 40369.217178
validation loss after epoch 4 : 3091.989666
	Epoch 5....
validAcc: 0.095
Epoch has taken 0:11:30.246031
Number of used sentences in train = 9526
Total loss for epoch 5: 38357.238724
validation loss after epoch 5 : 2437.356698
validAcc: 0.746
	Epoch 6....
Epoch has taken 0:11:36.598062
Number of used sentences in train = 9526
Total loss for epoch 6: 36882.836714
validation loss after epoch 6 : 3635.130281
	Epoch 7....
validAcc: 0.247
Epoch has taken 0:11:34.475419
Number of used sentences in train = 9526
Total loss for epoch 7: 35698.936583
validation loss after epoch 7 : 2606.977335
validAcc: 0.755
	Epoch 8....
Epoch has taken 0:11:34.926954
Number of used sentences in train = 9526
Total loss for epoch 8: 34784.040695
validation loss after epoch 8 : 3835.587602
	Epoch 9....
validAcc: 0.121
Epoch has taken 0:11:54.669728
Number of used sentences in train = 9526
Total loss for epoch 9: 34012.808703
validation loss after epoch 9 : 2445.009334
	Epoch 10....
validAcc: 0.369
Epoch has taken 0:11:46.296290
Number of used sentences in train = 9526
Total loss for epoch 10: 33347.918819
validation loss after epoch 10 : 2779.116078
	Epoch 11....
validAcc: 0.114
Epoch has taken 0:11:34.606245
Number of used sentences in train = 9526
Total loss for epoch 11: 32828.008231
validation loss after epoch 11 : 2300.455130
	TransitionClassifier(
  (p_embeddings): Embedding(21, 40)
  (w_embeddings): Embedding(4470, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
validAcc: 0.142
Epoch has taken 0:11:30.573170
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/0.kiperwasser.p
Number of used sentences in train = 1059
Total loss for epoch 0: 5592.810311
	Epoch 1....
Epoch has taken 0:01:10.793811
Number of used sentences in train = 1059
Total loss for epoch 1: 2903.567425
	Epoch 2....
Epoch has taken 0:01:10.755201
Number of used sentences in train = 1059
Total loss for epoch 2: 1996.608525
	Epoch 3....
Epoch has taken 0:01:10.775218
Number of used sentences in train = 1059
Total loss for epoch 3: 1517.662401
	Epoch 4....
Epoch has taken 0:01:10.764524
Number of used sentences in train = 1059
Total loss for epoch 4: 1232.298676
	Epoch 5....
Epoch has taken 0:01:10.797591
Number of used sentences in train = 1059
Total loss for epoch 5: 1029.269479
	Epoch 6....
Epoch has taken 0:01:10.972515
Number of used sentences in train = 1059
Total loss for epoch 6: 921.379239
	Epoch 7....
Epoch has taken 0:01:10.819098
Number of used sentences in train = 1059
Total loss for epoch 7: 826.786912
	Epoch 8....
Epoch has taken 0:01:10.775514
Number of used sentences in train = 1059
Total loss for epoch 8: 762.292566
	Epoch 9....
Epoch has taken 0:01:10.759019
Number of used sentences in train = 1059
Total loss for epoch 9: 706.393709
	Epoch 10....
Epoch has taken 0:01:10.764057
Number of used sentences in train = 1059
Total loss for epoch 10: 657.316391
	Epoch 11....
Epoch has taken 0:01:10.719919
Number of used sentences in train = 1059
Total loss for epoch 11: 638.956281
Epoch has taken 0:01:10.754425

==================================================================================================
	Training time : 2:33:28.182357
==================================================================================================
	Identification : 0.015

*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|
	Language : EN
==================================================================================================
	Training (Important) : 11506, Test : 2541
	MWEs in tain : 7100, occurrences : 25744
	Impotant words in tain : 4638
	MWE length mean : 2.66
	Seen MWEs : 3130 (77 %)
	New MWEs : 909 (22 %)
==================================================================================================
	TransitionClassifier(
  (p_embeddings): Embedding(21, 40)
  (w_embeddings): Embedding(4640, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/67.kiperwasser.p
Number of used sentences in train = 10355
Total loss for epoch 0: 77533.728913
validation loss after epoch 0 : 7691.639213
	Epoch 1....
validAcc: 0.733
Epoch has taken 0:12:34.466376
Number of used sentences in train = 10355
Total loss for epoch 1: 58873.560477
validation loss after epoch 1 : 5025.440547
validAcc: 0.745
	Epoch 2....
Epoch has taken 0:12:33.057786
Number of used sentences in train = 10355
Total loss for epoch 2: 52430.295216
validation loss after epoch 2 : 4882.287458
validAcc: 0.747
	Epoch 3....
Epoch has taken 0:12:37.087604
Number of used sentences in train = 10355
Total loss for epoch 3: 48348.805966
validation loss after epoch 3 : 4683.455531
	Epoch 4....
validAcc: 0.746
Epoch has taken 0:12:33.822726
Number of used sentences in train = 10355
Total loss for epoch 4: 45611.042689
validation loss after epoch 4 : 4582.898858
validAcc: 0.759
	Epoch 5....
Epoch has taken 0:12:38.072038
Number of used sentences in train = 10355
Total loss for epoch 5: 43478.073922
validation loss after epoch 5 : 4582.176325
validAcc: 0.776
	Epoch 6....
Epoch has taken 0:12:40.935798
Number of used sentences in train = 10355
Total loss for epoch 6: 41855.215753
validation loss after epoch 6 : 4460.317188
	Epoch 7....
validAcc: 0.76
Epoch has taken 0:12:37.632617
Number of used sentences in train = 10355
Total loss for epoch 7: 40479.395643
validation loss after epoch 7 : 4577.735938
	Epoch 8....
validAcc: 0.762
Epoch has taken 0:12:40.143663
## OAR [2018-09-30 01:06:11] Job 1693763 KILLED ##
INFO: activate-binutils_linux-64.sh made the following environmental changes:
+ADDR2LINE=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-addr2line
+AR=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-ar
+AS=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-as
+CXXFILT=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-c++filt
+ELFEDIT=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-elfedit
+GPROF=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-gprof
+HOST=x86_64-conda_cos6-linux-gnu
+LD_GOLD=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-ld.gold
+LD=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-ld
+NM=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-nm
+OBJCOPY=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-objcopy
+OBJDUMP=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-objdump
+RANLIB=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-ranlib
+READELF=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-readelf
+SIZE=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-size
+STRINGS=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-strings
+STRIP=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-strip
INFO: activate-gcc_linux-64.sh made the following environmental changes:
+CC=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-cc
+CFLAGS=-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -pipe
+_CONDA_PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata_x86_64_conda_cos6_linux_gnu
+CPPFLAGS=-DNDEBUG -D_FORTIFY_SOURCE=2 -O2
+CPP=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-cpp
+DEBUG_CFLAGS=-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-all -fno-plt -Og -g -Wall -Wextra -fvar-tracking-assignments -pipe
+DEBUG_CPPFLAGS=-D_DEBUG -D_FORTIFY_SOURCE=2 -Og
+GCC_AR=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-gcc-ar
+GCC=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-gcc
+GCC_NM=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-gcc-nm
+GCC_RANLIB=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-gcc-ranlib
+LDFLAGS=-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now
INFO: activate-gxx_linux-64.sh made the following environmental changes:
+CXXFLAGS=-fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -pipe
+CXX=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-c++
+DEBUG_CXXFLAGS=-fvisibility-inlines-hidden -std=c++17 -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-all -fno-plt -Og -g -Wall -Wextra -fvar-tracking-assignments -pipe
+GXX=/home/halsaied/miniconda2/bin/x86_64-conda_cos6-linux-gnu-g++
	TransitionClassifier(
  (p_embeddings): Embedding(18, 40)
  (w_embeddings): Embedding(2944, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
4232 9887
837 1952
	Language : EN
==================================================================================================
	Training (Important) : 2643, Test : 1000
	MWEs in tain : 2869, occurrences : 4232
	Impotant words in tain : 2942
	MWE length mean : 2.34
	Seen MWEs : 187 (22 %)
	New MWEs : 650 (77 %)
==================================================================================================
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/423.kiperwasser.p
Number of used sentences in train = 2378
Total loss for epoch 0: 17659.085601
validation loss after epoch 0 : 1322.474161
	Epoch 1....
validAcc: 0.022
Epoch has taken 0:01:38.626590
Number of used sentences in train = 2378
Total loss for epoch 1: 12724.061183
validation loss after epoch 1 : 625.675426
validAcc: 0.053
	Epoch 2....
Epoch has taken 0:01:46.078715
Number of used sentences in train = 2378
Total loss for epoch 2: 10475.317263
validation loss after epoch 2 : 525.821882
validAcc: 0.633
	Epoch 3....
Epoch has taken 0:01:44.792818
Number of used sentences in train = 2378
Total loss for epoch 3: 8973.689250
validation loss after epoch 3 : 740.819060
	Epoch 4....
validAcc: 0.63
Epoch has taken 0:01:46.346707
Number of used sentences in train = 2378
Total loss for epoch 4: 7975.820955
validation loss after epoch 4 : 749.990945
	Epoch 5....
validAcc: 0.532
Epoch has taken 0:01:46.718513
Number of used sentences in train = 2378
Total loss for epoch 5: 7286.081728
validation loss after epoch 5 : 650.417884
	Epoch 6....
validAcc: 0.601
Epoch has taken 0:01:40.227118
Number of used sentences in train = 2378
Total loss for epoch 6: 6788.768425
validation loss after epoch 6 : 593.454344
	Epoch 7....
validAcc: 0.2
Epoch has taken 0:01:41.345081
Number of used sentences in train = 2378
Total loss for epoch 7: 6453.507279
validation loss after epoch 7 : 441.022856
	Epoch 8....
validAcc: 0.608
Epoch has taken 0:01:46.304721
Number of used sentences in train = 2378
Total loss for epoch 8: 6226.766196
validation loss after epoch 8 : 590.242858
	Epoch 9....
validAcc: 0.607
Epoch has taken 0:01:40.159728
Number of used sentences in train = 2378
Total loss for epoch 9: 6046.031030
validation loss after epoch 9 : 556.444495
	Epoch 10....
validAcc: 0.075
Epoch has taken 0:01:45.703625
Number of used sentences in train = 2378
Total loss for epoch 10: 5913.571598
validation loss after epoch 10 : 390.611801
	Epoch 11....
validAcc: 0.594
Epoch has taken 0:01:46.517757
Number of used sentences in train = 2378
Total loss for epoch 11: 5841.481590
validation loss after epoch 11 : 580.379665
	Epoch 12....
validAcc: 0.609
Epoch has taken 0:01:39.053281
Number of used sentences in train = 2378
Total loss for epoch 12: 5750.440534
validation loss after epoch 12 : 555.202085
	Epoch 13....
validAcc: 0.57
Epoch has taken 0:01:38.932611
Number of used sentences in train = 2378
Total loss for epoch 13: 5702.075042
validation loss after epoch 13 : 570.944098
	Epoch 14....
validAcc: 0.595
Epoch has taken 0:01:39.458592
Number of used sentences in train = 2378
Total loss for epoch 14: 5633.530669
validation loss after epoch 14 : 515.692407
	TransitionClassifier(
  (p_embeddings): Embedding(18, 40)
  (w_embeddings): Embedding(2944, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
validAcc: 0.6
Epoch has taken 0:01:38.455231
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/0.kiperwasser.p
Number of used sentences in train = 265
Total loss for epoch 0: 1479.340279
	Epoch 1....
Epoch has taken 0:00:08.550412
Number of used sentences in train = 265
Total loss for epoch 1: 909.056852
	Epoch 2....
Epoch has taken 0:00:08.547635
Number of used sentences in train = 265
Total loss for epoch 2: 696.408779
	Epoch 3....
Epoch has taken 0:00:08.539036
Number of used sentences in train = 265
Total loss for epoch 3: 595.623603
	Epoch 4....
Epoch has taken 0:00:08.563806
Number of used sentences in train = 265
Total loss for epoch 4: 545.964592
	Epoch 5....
Epoch has taken 0:00:08.560220
Number of used sentences in train = 265
Total loss for epoch 5: 528.301437
	Epoch 6....
Epoch has taken 0:00:08.524749
Number of used sentences in train = 265
Total loss for epoch 6: 510.966137
	Epoch 7....
Epoch has taken 0:00:08.541071
Number of used sentences in train = 265
Total loss for epoch 7: 501.477319
	Epoch 8....
Epoch has taken 0:00:08.549643
Number of used sentences in train = 265
Total loss for epoch 8: 494.744820
	Epoch 9....
Epoch has taken 0:00:08.555179
Number of used sentences in train = 265
Total loss for epoch 9: 484.317965
	Epoch 10....
Epoch has taken 0:00:08.574979
Number of used sentences in train = 265
Total loss for epoch 10: 479.746202
	Epoch 11....
Epoch has taken 0:00:08.539782
Number of used sentences in train = 265
Total loss for epoch 11: 473.103858
	Epoch 12....
Epoch has taken 0:00:08.529020
Number of used sentences in train = 265
Total loss for epoch 12: 466.530370
	Epoch 13....
Epoch has taken 0:00:08.558161
Number of used sentences in train = 265
Total loss for epoch 13: 464.652345
	Epoch 14....
Epoch has taken 0:00:08.553529
Number of used sentences in train = 265
Total loss for epoch 14: 462.140716
Epoch has taken 0:00:08.526314

==================================================================================================
	Training time : 0:28:00.464691
==================================================================================================
	Identification : 0.087

*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|
4232 9887
837 1952
	Language : EN
==================================================================================================
	Training (Important) : 2643, Test : 1000
	MWEs in tain : 2869, occurrences : 4232
	Impotant words in tain : 2942
	MWE length mean : 2.34
	Seen MWEs : 187 (22 %)
	New MWEs : 650 (77 %)
==================================================================================================
	TransitionClassifier(
  (p_embeddings): Embedding(18, 40)
  (w_embeddings): Embedding(2944, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/332.kiperwasser.p
Number of used sentences in train = 2378
Total loss for epoch 0: 17056.988169
validation loss after epoch 0 : 1708.201981
	Epoch 1....
validAcc: 0.223
Epoch has taken 0:01:39.035686
Number of used sentences in train = 2378
Total loss for epoch 1: 12488.115072
validation loss after epoch 1 : 796.195565
validAcc: 0.584
	Epoch 2....
Epoch has taken 0:01:40.051789
Number of used sentences in train = 2378
Total loss for epoch 2: 10312.137630
validation loss after epoch 2 : 986.474489
validAcc: 0.61
	Epoch 3....
Epoch has taken 0:01:39.535866
Number of used sentences in train = 2378
Total loss for epoch 3: 8901.538193
validation loss after epoch 3 : 898.271149
	Epoch 4....
validAcc: 0.572
Epoch has taken 0:01:39.035961
Number of used sentences in train = 2378
Total loss for epoch 4: 7854.260781
validation loss after epoch 4 : 823.330259
	Epoch 5....
validAcc: 0.098
Epoch has taken 0:01:38.240716
Number of used sentences in train = 2378
Total loss for epoch 5: 7199.393827
validation loss after epoch 5 : 538.628552
	Epoch 6....
validAcc: 0.59
Epoch has taken 0:01:39.203611
Number of used sentences in train = 2378
Total loss for epoch 6: 6702.489739
validation loss after epoch 6 : 742.561324
	Epoch 7....
validAcc: 0.606
Epoch has taken 0:01:38.667139
Number of used sentences in train = 2378
Total loss for epoch 7: 6348.375238
validation loss after epoch 7 : 699.301962
	Epoch 8....
validAcc: 0.135
Epoch has taken 0:01:39.157694
Number of used sentences in train = 2378
Total loss for epoch 8: 6102.368635
validation loss after epoch 8 : 499.267905
	Epoch 9....
validAcc: 0.6
Epoch has taken 0:01:38.670560
Number of used sentences in train = 2378
Total loss for epoch 9: 5947.721392
validation loss after epoch 9 : 687.493728
	Epoch 10....
validAcc: 0.279
Epoch has taken 0:01:39.681718
Number of used sentences in train = 2378
Total loss for epoch 10: 5819.733165
validation loss after epoch 10 : 494.580974
	Epoch 11....
validAcc: 0.583
Epoch has taken 0:01:38.575898
Number of used sentences in train = 2378
Total loss for epoch 11: 5736.254729
validation loss after epoch 11 : 647.219145
	Epoch 12....
validAcc: 0.25
Epoch has taken 0:01:39.213891
Number of used sentences in train = 2378
Total loss for epoch 12: 5664.362730
validation loss after epoch 12 : 479.043217
	Epoch 13....
validAcc: 0.607
Epoch has taken 0:01:38.595960
Number of used sentences in train = 2378
Total loss for epoch 13: 5598.452252
validation loss after epoch 13 : 628.886477
	Epoch 14....
validAcc: 0.578
Epoch has taken 0:01:39.397081
Number of used sentences in train = 2378
Total loss for epoch 14: 5572.570362
validation loss after epoch 14 : 605.541525
	TransitionClassifier(
  (p_embeddings): Embedding(18, 40)
  (w_embeddings): Embedding(2944, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
validAcc: 0.198
Epoch has taken 0:01:38.651671
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/0.kiperwasser.p
Number of used sentences in train = 265
Total loss for epoch 0: 1416.859472
	Epoch 1....
Epoch has taken 0:00:09.767301
Number of used sentences in train = 265
Total loss for epoch 1: 796.864744
	Epoch 2....
Epoch has taken 0:00:09.793305
Number of used sentences in train = 265
Total loss for epoch 2: 455.839904
	Epoch 3....
Epoch has taken 0:00:09.742279
Number of used sentences in train = 265
Total loss for epoch 3: 327.165605
	Epoch 4....
Epoch has taken 0:00:09.772833
Number of used sentences in train = 265
Total loss for epoch 4: 262.944402
	Epoch 5....
Epoch has taken 0:00:09.748872
Number of used sentences in train = 265
Total loss for epoch 5: 249.958484
	Epoch 6....
Epoch has taken 0:00:09.785837
Number of used sentences in train = 265
Total loss for epoch 6: 217.287016
	Epoch 7....
Epoch has taken 0:00:09.802442
Number of used sentences in train = 265
Total loss for epoch 7: 193.489760
	Epoch 8....
Epoch has taken 0:00:09.759056
Number of used sentences in train = 265
Total loss for epoch 8: 182.600091
	Epoch 9....
Epoch has taken 0:00:09.782824
Number of used sentences in train = 265
Total loss for epoch 9: 175.688057
	Epoch 10....
Epoch has taken 0:00:09.758044
Number of used sentences in train = 265
Total loss for epoch 10: 171.236826
	Epoch 11....
Epoch has taken 0:00:09.804948
Number of used sentences in train = 265
Total loss for epoch 11: 169.441389
	Epoch 12....
Epoch has taken 0:00:09.803148
Number of used sentences in train = 265
Total loss for epoch 12: 166.663692
	Epoch 13....
Epoch has taken 0:00:09.773884
Number of used sentences in train = 265
Total loss for epoch 13: 165.216442
	Epoch 14....
Epoch has taken 0:00:09.793965
Number of used sentences in train = 265
Total loss for epoch 14: 159.010101
Epoch has taken 0:00:09.742297

==================================================================================================
	Training time : 0:27:12.636823
==================================================================================================
	Identification : 0.024

*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|
	Language : FR
==================================================================================================
	Training (Important) : 11506, Test : 2541
	MWEs in tain : 7100, occurrences : 25744
	Impotant words in tain : 4638
	MWE length mean : 2.66
	Seen MWEs : 3130 (77 %)
	New MWEs : 909 (22 %)
==================================================================================================
	TransitionClassifier(
  (p_embeddings): Embedding(21, 40)
  (w_embeddings): Embedding(4640, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/294.kiperwasser.p
Number of used sentences in train = 10355
Total loss for epoch 0: 73728.342931
validation loss after epoch 0 : 7485.090236
	Epoch 1....
validAcc: 0.082
Epoch has taken 0:12:42.500408
Number of used sentences in train = 10355
Total loss for epoch 1: 55890.962379
validation loss after epoch 1 : 4312.572510
validAcc: 0.114
	Epoch 2....
Epoch has taken 0:12:48.202487
Number of used sentences in train = 10355
Total loss for epoch 2: 49846.716816
validation loss after epoch 2 : 3829.796494
validAcc: 0.753
	Epoch 3....
Epoch has taken 0:12:37.744148
Number of used sentences in train = 10355
Total loss for epoch 3: 46131.010087
validation loss after epoch 3 : 4713.113239
	Epoch 4....
validAcc: 0.623
Epoch has taken 0:12:55.955772
Number of used sentences in train = 10355
Total loss for epoch 4: 43487.392257
validation loss after epoch 4 : 4021.667699
	Epoch 5....
validAcc: 0.109
Epoch has taken 0:12:50.933535
Number of used sentences in train = 10355
Total loss for epoch 5: 41586.828445
validation loss after epoch 5 : 2949.135655
	Epoch 6....
validAcc: 0.113
Epoch has taken 0:12:52.699015
Number of used sentences in train = 10355
Total loss for epoch 6: 40195.199713
validation loss after epoch 6 : 2893.182169
validAcc: 0.781
	Epoch 7....
Epoch has taken 0:12:51.896781
Number of used sentences in train = 10355
Total loss for epoch 7: 39088.361660
validation loss after epoch 7 : 4688.417545
	Epoch 8....
validAcc: 0.144
Epoch has taken 0:12:40.628056
Number of used sentences in train = 10355
Total loss for epoch 8: 38211.040940
validation loss after epoch 8 : 2811.881263
	Epoch 9....
validAcc: 0.359
Epoch has taken 0:12:50.256777
Number of used sentences in train = 10355
Total loss for epoch 9: 37429.546967
validation loss after epoch 9 : 3164.624065
	Epoch 10....
validAcc: 0.669
Epoch has taken 0:12:43.756155
Number of used sentences in train = 10355
Total loss for epoch 10: 36743.371801
validation loss after epoch 10 : 4055.973337
	Epoch 11....
validAcc: 0.753
Epoch has taken 0:12:49.845442
Number of used sentences in train = 10355
Total loss for epoch 11: 36136.660325
validation loss after epoch 11 : 4221.107478
	Epoch 12....
validAcc: 0.367
Epoch has taken 0:12:40.428870
Number of used sentences in train = 10355
Total loss for epoch 12: 35581.842314
validation loss after epoch 12 : 3167.338566
	Epoch 13....
validAcc: 0.111
Epoch has taken 0:12:41.833864
Number of used sentences in train = 10355
Total loss for epoch 13: 35228.511643
validation loss after epoch 13 : 2732.090322
	Epoch 14....
validAcc: 0.1
Epoch has taken 0:12:56.404892
Number of used sentences in train = 10355
Total loss for epoch 14: 34830.170325
validation loss after epoch 14 : 2602.053964
	TransitionClassifier(
  (p_embeddings): Embedding(21, 40)
  (w_embeddings): Embedding(4640, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
validAcc: 0.11
Epoch has taken 0:12:38.547379
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/0.kiperwasser.p
Number of used sentences in train = 1151
Total loss for epoch 0: 6897.711782
	Epoch 1....
Epoch has taken 0:01:17.302935
Number of used sentences in train = 1151
Total loss for epoch 1: 3453.898752
	Epoch 2....
Epoch has taken 0:01:17.321969
Number of used sentences in train = 1151
Total loss for epoch 2: 2250.668305
	Epoch 3....
Epoch has taken 0:01:17.260831
Number of used sentences in train = 1151
Total loss for epoch 3: 1597.608299
	Epoch 4....
Epoch has taken 0:01:17.316678
Number of used sentences in train = 1151
Total loss for epoch 4: 1239.759807
	Epoch 5....
Epoch has taken 0:01:17.366999
Number of used sentences in train = 1151
Total loss for epoch 5: 996.491218
	Epoch 6....
Epoch has taken 0:01:17.347423
Number of used sentences in train = 1151
Total loss for epoch 6: 860.959082
	Epoch 7....
Epoch has taken 0:01:17.352402
Number of used sentences in train = 1151
Total loss for epoch 7: 760.979406
	Epoch 8....
Epoch has taken 0:01:17.302282
Number of used sentences in train = 1151
Total loss for epoch 8: 700.270188
	Epoch 9....
Epoch has taken 0:01:17.325013
Number of used sentences in train = 1151
Total loss for epoch 9: 654.749175
	Epoch 10....
Epoch has taken 0:01:17.277603
Number of used sentences in train = 1151
Total loss for epoch 10: 610.058193
	Epoch 11....
Epoch has taken 0:01:17.319412
Number of used sentences in train = 1151
Total loss for epoch 11: 586.821483
	Epoch 12....
Epoch has taken 0:01:17.483003
Number of used sentences in train = 1151
Total loss for epoch 12: 548.264838
	Epoch 13....
Epoch has taken 0:01:17.376327
Number of used sentences in train = 1151
Total loss for epoch 13: 520.948296
	Epoch 14....
Epoch has taken 0:01:17.266887
Number of used sentences in train = 1151
Total loss for epoch 14: 500.309489
Epoch has taken 0:01:17.960222

==================================================================================================
	Training time : 3:31:04.375724
==================================================================================================
	Identification : 0.012

*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|
	Language : FR
==================================================================================================
	Training (Important) : 10585, Test : 2541
	MWEs in tain : 6789, occurrences : 23628
	Impotant words in tain : 4468
	MWE length mean : 2.66
	Seen MWEs : 3102 (76 %)
	New MWEs : 937 (23 %)
==================================================================================================
	TransitionClassifier(
  (p_embeddings): Embedding(21, 40)
  (w_embeddings): Embedding(4470, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/265.kiperwasser.p
Number of used sentences in train = 9526
Total loss for epoch 0: 73660.820108
validation loss after epoch 0 : 6756.440522
	Epoch 1....
validAcc: 0.683
Epoch has taken 0:11:45.327370
Number of used sentences in train = 9526
Total loss for epoch 1: 54305.192163
validation loss after epoch 1 : 4470.693680
validAcc: 0.731
	Epoch 2....
Epoch has taken 0:11:47.080136
Number of used sentences in train = 9526
Total loss for epoch 2: 48073.262286
validation loss after epoch 2 : 4385.042482
	Epoch 3....
validAcc: 0.705
Epoch has taken 0:11:42.747425
Number of used sentences in train = 9526
Total loss for epoch 3: 44229.770275
validation loss after epoch 3 : 4014.241838
validAcc: 0.743
	Epoch 4....
Epoch has taken 0:11:43.442988
Number of used sentences in train = 9526
Total loss for epoch 4: 41462.171466
validation loss after epoch 4 : 4354.851022
validAcc: 0.75
	Epoch 5....
Epoch has taken 0:11:48.005900
Number of used sentences in train = 9526
Total loss for epoch 5: 39401.410574
validation loss after epoch 5 : 4103.311597
	Epoch 6....
validAcc: 0.745
Epoch has taken 0:12:18.845599
Number of used sentences in train = 9526
Total loss for epoch 6: 37869.800637
validation loss after epoch 6 : 4015.681811
	Epoch 7....
validAcc: 0.744
Epoch has taken 0:11:50.174841
Number of used sentences in train = 9526
Total loss for epoch 7: 36620.189309
validation loss after epoch 7 : 3781.437416
	Epoch 8....
validAcc: 0.138
Epoch has taken 0:11:42.522822
Number of used sentences in train = 9526
Total loss for epoch 8: 35606.266554
validation loss after epoch 8 : 2408.667147
	Epoch 9....
validAcc: 0.206
Epoch has taken 0:11:47.225815
Number of used sentences in train = 9526
Total loss for epoch 9: 34775.350607
validation loss after epoch 9 : 2528.562423
	Epoch 10....
validAcc: 0.082
Epoch has taken 0:11:43.869405
Number of used sentences in train = 9526
Total loss for epoch 10: 34185.443251
validation loss after epoch 10 : 2328.444249
	Epoch 11....
validAcc: 0.29
Epoch has taken 0:12:03.595799
Number of used sentences in train = 9526
Total loss for epoch 11: 33599.267822
validation loss after epoch 11 : 2719.902351
validAcc: 0.752
	Epoch 12....
Epoch has taken 0:12:27.664029
Number of used sentences in train = 9526
Total loss for epoch 12: 33148.904261
validation loss after epoch 12 : 3740.799196
	Epoch 13....
validAcc: 0.121
Epoch has taken 0:12:13.888771
Number of used sentences in train = 9526
Total loss for epoch 13: 32749.292546
validation loss after epoch 13 : 2453.778890
	Epoch 14....
validAcc: 0.147
Epoch has taken 0:11:43.356948
Number of used sentences in train = 9526
Total loss for epoch 14: 32470.237078
validation loss after epoch 14 : 2479.626021
	TransitionClassifier(
  (p_embeddings): Embedding(21, 40)
  (w_embeddings): Embedding(4470, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
validAcc: 0.17
Epoch has taken 0:12:07.626465
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/0.kiperwasser.p
Number of used sentences in train = 1059
Total loss for epoch 0: 6635.797204
	Epoch 1....
Epoch has taken 0:01:11.728118
Number of used sentences in train = 1059
Total loss for epoch 1: 3794.451383
	Epoch 2....
Epoch has taken 0:01:12.304596
Number of used sentences in train = 1059
Total loss for epoch 2: 2776.627658
	Epoch 3....
Epoch has taken 0:01:12.228194
Number of used sentences in train = 1059
Total loss for epoch 3: 2081.243902
	Epoch 4....
Epoch has taken 0:01:12.360372
Number of used sentences in train = 1059
Total loss for epoch 4: 1663.132868
	Epoch 5....
Epoch has taken 0:01:12.350832
Number of used sentences in train = 1059
Total loss for epoch 5: 1373.349157
	Epoch 6....
Epoch has taken 0:01:12.302795
Number of used sentences in train = 1059
Total loss for epoch 6: 1170.085492
	Epoch 7....
Epoch has taken 0:01:12.317995
Number of used sentences in train = 1059
Total loss for epoch 7: 1028.335178
	Epoch 8....
Epoch has taken 0:01:12.317681
Number of used sentences in train = 1059
Total loss for epoch 8: 931.132564
	Epoch 9....
Epoch has taken 0:01:12.316927
Number of used sentences in train = 1059
Total loss for epoch 9: 868.687992
	Epoch 10....
Epoch has taken 0:01:12.319539
Number of used sentences in train = 1059
Total loss for epoch 10: 822.010308
	Epoch 11....
Epoch has taken 0:01:12.273709
Number of used sentences in train = 1059
Total loss for epoch 11: 782.022207
	Epoch 12....
Epoch has taken 0:01:18.569779
Number of used sentences in train = 1059
Total loss for epoch 12: 742.832886
	Epoch 13....
Epoch has taken 0:01:12.344708
Number of used sentences in train = 1059
Total loss for epoch 13: 716.556196
	Epoch 14....
Epoch has taken 0:01:12.293440
Number of used sentences in train = 1059
Total loss for epoch 14: 686.747667
Epoch has taken 0:01:11.983266

==================================================================================================
	Training time : 3:16:57.377360
==================================================================================================
	Identification : 0.019

*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|
	Language : FR
==================================================================================================
	Training (Important) : 10585, Test : 1235
	MWEs in tain : 6789, occurrences : 23628
	Impotant words in tain : 4468
	MWE length mean : 2.66
	Seen MWEs : 1753 (82 %)
	New MWEs : 363 (17 %)
==================================================================================================
	TransitionClassifier(
  (p_embeddings): Embedding(21, 40)
  (w_embeddings): Embedding(4470, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/147.kiperwasser.p
Number of used sentences in train = 9526
Total loss for epoch 0: 72686.063418
validation loss after epoch 0 : 6681.470024
	Epoch 1....
validAcc: 0.703
Epoch has taken 0:11:42.788043
Number of used sentences in train = 9526
Total loss for epoch 1: 53807.801017
validation loss after epoch 1 : 5501.684372
	Epoch 2....
validAcc: 0.674
Epoch has taken 0:11:41.963481
Number of used sentences in train = 9526
Total loss for epoch 2: 47562.991333
validation loss after epoch 2 : 4447.413535
validAcc: 0.743
	Epoch 3....
Epoch has taken 0:11:45.030662
Number of used sentences in train = 9526
Total loss for epoch 3: 43856.162291
validation loss after epoch 3 : 4178.177763
	Epoch 4....
validAcc: 0.718
Epoch has taken 0:11:44.124943
Number of used sentences in train = 9526
Total loss for epoch 4: 41448.550302
validation loss after epoch 4 : 4151.696576
	Epoch 5....
validAcc: 0.741
Epoch has taken 0:11:49.795607
Number of used sentences in train = 9526
Total loss for epoch 5: 39671.137817
validation loss after epoch 5 : 3788.624241
validAcc: 0.744
	Epoch 6....
Epoch has taken 0:12:17.482733
Number of used sentences in train = 9526
Total loss for epoch 6: 38224.520045
validation loss after epoch 6 : 3724.636989
	Epoch 7....
validAcc: 0.018
Epoch has taken 0:11:50.076519
Number of used sentences in train = 9526
Total loss for epoch 7: 36982.970437
validation loss after epoch 7 : 2410.491707
validAcc: 0.75
	Epoch 8....
Epoch has taken 0:11:46.626528
Number of used sentences in train = 9526
Total loss for epoch 8: 36116.277541
validation loss after epoch 8 : 3708.953717
	Epoch 9....
validAcc: 0.029
Epoch has taken 0:11:43.670574
Number of used sentences in train = 9526
Total loss for epoch 9: 35245.431113
validation loss after epoch 9 : 2285.373345
	Epoch 10....
validAcc: 0.732
Epoch has taken 0:11:46.145940
Number of used sentences in train = 9526
Total loss for epoch 10: 34486.421512
validation loss after epoch 10 : 3649.086509
	Epoch 11....
validAcc: 0.732
Epoch has taken 0:11:41.491762
Number of used sentences in train = 9526
Total loss for epoch 11: 33952.975812
validation loss after epoch 11 : 3392.537320
	Epoch 12....
validAcc: 0.737
Epoch has taken 0:11:40.867972
Number of used sentences in train = 9526
Total loss for epoch 12: 33473.187052
validation loss after epoch 12 : 3489.136899
	Epoch 13....
validAcc: 0.733
Epoch has taken 0:11:45.069160
Number of used sentences in train = 9526
Total loss for epoch 13: 33003.456918
validation loss after epoch 13 : 3502.673760
	Epoch 14....
validAcc: 0.742
Epoch has taken 0:11:51.199690
Number of used sentences in train = 9526
Total loss for epoch 14: 32629.464200
validation loss after epoch 14 : 3521.301442
	TransitionClassifier(
  (p_embeddings): Embedding(21, 40)
  (w_embeddings): Embedding(4470, 240)
  (lstm): LSTM(280, 90, bidirectional=True)
  (linear1): Linear(in_features=1440, out_features=11, bias=True)
  (linear2): Linear(in_features=11, out_features=4, bias=True)
)
==================================================================================================
	Adagrad (
Parameter Group 0
    initial_accumulator_value: 0
    lr: 0.07
    lr_decay: 0
    weight_decay: 0
)
==================================================================================================
	NLLLoss()
==================================================================================================
	Epoch 0....
validAcc: 0.741
Epoch has taken 0:11:49.586776
# Network optimizer = Adagrad, learning rate = 0.07

/home/halsaied/NNIdenSys/Reports/0.kiperwasser.p
Number of used sentences in train = 1059
Total loss for epoch 0: 8027.005701
	Epoch 1....
Epoch has taken 0:01:14.197785
Number of used sentences in train = 1059
Total loss for epoch 1: 5316.587838
	Epoch 2....
Epoch has taken 0:01:14.129136
Number of used sentences in train = 1059
Total loss for epoch 2: 4378.632385
	Epoch 3....
Epoch has taken 0:01:14.178671
Number of used sentences in train = 1059
Total loss for epoch 3: 3901.138740
	Epoch 4....
Epoch has taken 0:01:14.196204
Number of used sentences in train = 1059
Total loss for epoch 4: 3635.839606
	Epoch 5....
Epoch has taken 0:01:14.195710
Number of used sentences in train = 1059
Total loss for epoch 5: 3506.200378
	Epoch 6....
Epoch has taken 0:01:14.200882
Number of used sentences in train = 1059
Total loss for epoch 6: 3417.188123
	Epoch 7....
Epoch has taken 0:01:14.215053
Number of used sentences in train = 1059
Total loss for epoch 7: 3329.249097
	Epoch 8....
Epoch has taken 0:01:14.196215
Number of used sentences in train = 1059
Total loss for epoch 8: 3266.301505
	Epoch 9....
Epoch has taken 0:01:14.197422
Number of used sentences in train = 1059
Total loss for epoch 9: 3229.191012
	Epoch 10....
Epoch has taken 0:01:14.305214
Number of used sentences in train = 1059
Total loss for epoch 10: 3203.503285
	Epoch 11....
Epoch has taken 0:01:14.352128
Number of used sentences in train = 1059
Total loss for epoch 11: 3182.104111
	Epoch 12....
Epoch has taken 0:01:14.385787
Number of used sentences in train = 1059
Total loss for epoch 12: 3162.089667
	Epoch 13....
Epoch has taken 0:01:14.353693
Number of used sentences in train = 1059
Total loss for epoch 13: 3142.924222
	Epoch 14....
Epoch has taken 0:01:14.418371
Number of used sentences in train = 1059
Total loss for epoch 14: 3118.847550
Epoch has taken 0:01:14.374409

==================================================================================================
	Training time : 3:15:31.799200
==================================================================================================
	Identification : 0.125

*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|*|
