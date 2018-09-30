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
