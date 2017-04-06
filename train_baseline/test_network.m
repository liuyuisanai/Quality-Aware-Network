for model_index=1:10
    caffe.reset_all;
    net=caffe.get_net(param.test_net_file,'test');
   % net.copy_from(strcat(param.save_model_file,num2str(split_index),'/',param.save_model_name,'_',num2str(2600),'.caffemodel'));
%     net.copy_from(param.fintune_model);
    tic;
    fin=fopen(param.result_save_file,'a');
    str=strcat(param.save_model_file,num2str(split_index),'/',param.save_model_name,'_',num2str(2600),'.caffemodel');
    fprintf(fin,'model:%s\n',str);
    fprintf('testing model:%s\n',str);
    fclose(fin);
    net.blobs('data').reshape([224 224 3 param.test_batch_size]);
    load(strcat(param.data_path,param.testdata_filename,num2str(split_index),'/test_data.mat'));
    test_script;
    
    
    toc;
end