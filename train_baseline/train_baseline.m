%%train network config
param.caffe_path=fullfile(fileparts(pwd),'matcaffe');
param.solver_netfile=fullfile(fileparts(pwd),'model','solver_bn.prototxt');
param.fintune_model=fullfile(fileparts(pwd),'pretrain_model','googlenet_bn.caffemodel');
param.test_net_file=fullfile(fileparts(pwd),'model','test_bn.prototxt');
param.test_batch_size=16;
param.result_save_file='result.txt';
param.save_model_file='model_cross';
param.save_model_name='base_train_cross';
param.test_interval=500;
param.train_maxiter=4000;
param.data_path=fullfile(fileparts(pwd),'generate_data');
param.traindata_filename='train_data_Prid';
param.testdata_filename='test_data_Prid';
param.use_data_split_index=1;
param.data_split_num=1;
param.train_person_num=150;
param.use_gpu=1;
param.gpu_id=0;
%%
for split_index=param.use_data_split_index:param.data_split_num
    if ~exist(strcat(param.save_model_file,num2str(split_index)),'file')
        mkdir(strcat(param.save_model_file,num2str(split_index)));
    end
    load(strcat(param.data_path,param.traindata_filename,num2str(split_index),'/train_data.mat'));
    %% find caffe
    cur_path=pwd;
    addpath(genpath(param.caffe_path))
    cd(param.caffe_path);
    caffe.reset_all;
    caffe.init_log(fullfile(cur_path,'log'));
    addpath(param.caffe_path);
    if param.use_gpu
        caffe.set_mode_gpu;
    else
        caffe.set_mode_cpu;
    end
    cd(cur_path);
    caffe_solver=caffe.get_solver(param.solver_netfile,param.gpu_id);
    net=caffe.get_net(param.test_net_file,'test');
    caffe_solver.use_caffemodel(param.fintune_model);
    input_data_shape=caffe_solver.nets{1}.blobs('data').shape;
    batch_size=input_data_shape(4);
    batch_data=zeros(input_data_shape,'single');
    batch_label=zeros(1,1,1,batch_size,'single');
    train_x_axis=[];
    train_y_axis=[];
    iter=0;
    while iter<param.train_maxiter
        
        [batch_data,batch_label]=get_train_minibatch(batch_data,batch_label,batch_size,train_data_cam1,train_data_cam2,label_train_cam1,label_train_cam2);
        caffe_solver.nets{1}.blobs('data').set_data(batch_data);
        caffe_solver.nets{1}.blobs('label').set_data(batch_label);
        caffe_solver.step(1);
        iter = caffe_solver.iter;
        if mod(iter,10)==0
            soft_loss=caffe_solver.nets{1}.blobs('loss3/loss').get_data;
            acc=caffe_solver.nets{1}.blobs('loss3/top-1').get_data;
            loss1=caffe_solver.nets{1}.blobs('loss1/loss').get_data;
            loss2=caffe_solver.nets{1}.blobs('loss2/loss').get_data;
%             loss4=caffe_solver.net.blobs('loss4').get_data;
            train_x_axis=[train_x_axis,iter];
            train_y_axis=[train_y_axis,soft_loss];
            plot(train_x_axis,train_y_axis);
            drawnow;
            fprintf('iter= %d ,softmaxloss= %f ,loss1=%f  loss2=%f\n',iter,soft_loss,loss1,loss2);
        end
        if mod(iter,param.test_interval)==0
            caffe_solver.nets{1}.save(strcat(param.save_model_file,num2str(split_index),'/',param.save_model_name,'_',num2str(iter),'.caffemodel'));
            net.copy_from(strcat(param.save_model_file,num2str(split_index),'/',param.save_model_name,'_',num2str(iter),'.caffemodel'));
            net.blobs('data').reshape([224 224 3 param.test_batch_size]);
            load(strcat(param.data_path,param.testdata_filename,num2str(split_index),'/test_data.mat'));
            test_script;
        end
        if iter>100 && mod(iter,100)==0
            caffe_solver.nets{1}.save(strcat(param.save_model_file,num2str(split_index),'/',param.save_model_name,'_',num2str(iter),'.caffemodel'));
        end
    end
%     test_network;
end