%%generate data config
param.split_data_num=1;
param.file_path_cam1=fullfile(fileparts(pwd),'prid_2011','multi_shot','cam_a');
param.file_path_cam2=fullfile(fileparts(pwd),'prid_2011','multi_shot','cam_b');
param.save_traindata_filename='train_data_Prid';
param.save_testdata_filename='test_data_Prid';
%% generate ten split train/test data
for split_num_start=1:param.split_data_num
    if ~exist(strcat(param.save_traindata_filename,num2str(split_num_start)),'file')
        mkdir(strcat(param.save_traindata_filename,num2str(split_num_start)));
    end
    if ~exist(strcat(param.save_testdata_filename,num2str(split_num_start)),'file')
        mkdir(strcat(param.save_testdata_filename,num2str(split_num_start)));
    end
    subdir_cam1=dir(param.file_path_cam1);
    subdir_cam2=dir(param.file_path_cam2);
    %% select images >27 frame
    person_file_cam1=[];
    person_file_cam2=[];
    for i=3:202
        image_path_a=fullfile(param.file_path_cam1,subdir_cam1(i).name,'/');
        image_list_a=dir(image_path_a);
        len=length(image_list_a);
        if len<29
            continue;
        else
            person_file_cam1=[person_file_cam1;i-2];        
        end
    end
    for i=3:202
        image_path_b=fullfile(param.file_path_cam2,subdir_cam2(i).name,'/');
        image_list_b=dir(image_path_b);
        len=length(image_list_b);
        if len<29 
            continue;
        else
            person_file_cam2=[person_file_cam2;i-2];
        end
    end
    %%
    person_num=intersect(person_file_cam1,person_file_cam2);
    select_list_len=size(person_num,1);
    train_data_select=randperm(select_list_len,select_list_len/2);
    train_data_index=person_num(train_data_select);
    test_data_index=setdiff(person_num,train_data_index);
    %generate train data
    train_data_cam1=[];
    label_train_cam1=[];
    train_image_name_cam1={};
    train_data_cam2=[];
    label_train_cam2=[];
    train_image_name_cam2={};
    for i=1:(length(train_data_index))
        fprintf('process train data:%d/%d\n',i,(length(train_data_index)));
        index=train_data_index(i);
        image_path_cam1=fullfile(param.file_path_cam1,subdir_cam1(index+2).name,'/');
        image_list_cam1=dir(image_path_cam1);
        for j=3:length(image_list_cam1);
            image_name=strcat(image_path_cam1,image_list_cam1(j).name);
            image_data=imread(image_name);
            train_data_cam1=[train_data_cam1;reshape(image_data,1,64*128*3)];
            label_train_cam1=[label_train_cam1;i];
            train_image_name_cam1=[train_image_name_cam1,image_name];
        end
        image_path_cam2=fullfile(param.file_path_cam2,subdir_cam2(index+2).name,'/');
        image_list_cam2=dir(image_path_cam2);
        for j=3:length(image_list_cam2);
            image_name=strcat(image_path_cam2,image_list_cam2(j).name);
            image_data=imread(image_name);
            train_data_cam2=[train_data_cam2;reshape(image_data,1,64*128*3)];
            label_train_cam2=[label_train_cam2;i];
            train_image_name_cam2=[train_image_name_cam2,image_name];
        end
    end
    save(strcat(param.save_traindata_filename,num2str(split_num_start),'/train_data.mat'),'train_data_cam1','train_data_cam2','label_train_cam1','label_train_cam2','train_image_name_cam1','train_image_name_cam2');
    %% generate test data
    test_data_cam1=[];
    label_test_cam1=[];
    test_image_name_cam1={};
    test_data_cam2=[];
    label_test_cam2=[];
    test_image_name_cam2={};
    for i=1:(length(test_data_index))
        fprintf('process test data:%d/%d\n',i,(length(test_data_index)));
        index=test_data_index(i);
        image_path_cam1=fullfile(param.file_path_cam1,subdir_cam1(index+2).name,'/');
        image_list_cam1=dir(image_path_cam1);
        for j=3:length(image_list_cam1);
            image_name=strcat(image_path_cam1,image_list_cam1(j).name);
            image_data=imread(image_name);
            test_data_cam1=[test_data_cam1;reshape(image_data,1,64*128*3)];
            label_test_cam1=[label_test_cam1;i+89];
            test_image_name_cam1=[test_image_name_cam1,image_name];
        end
        image_path_cam2=fullfile(param.file_path_cam2,subdir_cam2(index+2).name,'/');
        image_list_cam2=dir(image_path_cam2);
        for j=3:length(image_list_cam2);
            image_name=strcat(image_path_cam2,image_list_cam2(j).name);
            image_data=imread(image_name);
            test_data_cam2=[test_data_cam2;reshape(image_data,1,64*128*3)];
            label_test_cam2=[label_test_cam2;i+89];
            test_image_name_cam2=[test_image_name_cam2,image_name];
        end
    end
    save(strcat(param.save_testdata_filename,num2str(split_num_start),'/test_data.mat'),'test_data_cam1','test_data_cam2','label_test_cam1','label_test_cam2','test_image_name_cam1','test_image_name_cam2');
end