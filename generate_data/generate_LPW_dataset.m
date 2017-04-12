%%generate data config
param.split_data_num=1;
param.file_path_cam1=fullfile(fileparts(pwd),'people_tuples','scen1');%for test
param.file_path_cam2=fullfile(fileparts(pwd),'people_tuples','scen2');%for train
param.file_path_cam3=fullfile(fileparts(pwd),'people_tuples','scen3');%for train
param.save_traindata_filename='train_data_LPW';
param.save_testdata_filename='test_data_LPW';
%%generate ten split train/test data
for split_num_start=1:param.split_data_num
    if ~exist(strcat(param.save_traindata_filename,num2str(split_num_start)),'file')
        mkdir(strcat(param.save_traindata_filename,num2str(split_num_start)));
    end
    if ~exist(strcat(param.save_testdata_filename,num2str(split_num_start)),'file')
        mkdir(strcat(param.save_testdata_filename,num2str(split_num_start)));
    end
    subdir_cam3_view1=dir(strcat(param.file_path_cam3,'view1'));
    subdir_cam3_view2=dir(strcat(param.file_path_cam3,'view2'));
    subdir_cam3_view3=dir(strcat(param.file_path_cam3,'view3'));
    subdir_cam3_view4=dir(strcat(param.file_path_cam3,'view4'));
    subdir_cam2_view1=dir(strcat(param.file_path_cam2,'view1'));
    subdir_cam2_view2=dir(strcat(param.file_path_cam2,'view2'));
    subdir_cam2_view3=dir(strcat(param.file_path_cam2,'view3'));
    subdir_cam2_view4=dir(strcat(param.file_path_cam2,'view4'));
    
    subdir_cam1_view1=dir(strcat(param.file_path_cam1,'view1'));
    subdir_cam1_view2=dir(strcat(param.file_path_cam1,'view2'));
    subdir_cam1_view3=dir(strcat(param.file_path_cam1,'view3'));
    
    label_train_cam2_view1=[];
    train_image_name_cam2_view1={};
    label_train_cam2_view2=[];
    train_image_name_cam2_view2={};
    label_train_cam2_view3=[];
    train_image_name_cam2_view3={};
    label_train_cam2_view4=[];
    train_image_name_cam2_view4={};
    label_train_cam3_view1=[];
    train_image_name_cam3_view1={};
    label_train_cam3_view2=[];
    train_image_name_cam3_view2={};
    label_train_cam3_view3=[];
    train_image_name_cam3_view3={};
    label_train_cam3_view4=[];
    train_image_name_cam3_view4={};
    nameMapLabel_cam2=[];
    nameMapLabel_cam3=[];
    for i=3:(length(subdir_cam2_view3))
        fprintf('process test data cam2 view3:%d/%d\n',i,(length(subdir_cam2_view3)));
        image_path_cam2=strcat(param.file_path_cam2,'view3/',subdir_cam2_view3(i).name,'/');
        image_list_cam2=dir(image_path_cam2);
        nameMapLabel_cam2=[nameMapLabel_cam2;str2num(subdir_cam2_view3(i).name)];
        for j=3:length(image_list_cam2);
            image_name=strcat(image_path_cam2,image_list_cam2(j).name);
            label_train_cam2_view3=[label_train_cam2_view3;str2num(subdir_cam2_view3(i).name)];
            train_image_name_cam2_view3=[train_image_name_cam2_view3,image_name];
        end
    end
    for i=3:(length(subdir_cam2_view1))
        fprintf('process test data cam2 view1:%d/%d\n',i,(length(subdir_cam2_view1)));
        image_path_cam2=strcat(param.file_path_cam2,'view1/',subdir_cam2_view1(i).name,'/');
        image_list_cam2=dir(image_path_cam2);
        for j=3:length(image_list_cam2);
            image_name=strcat(image_path_cam2,image_list_cam2(j).name);
            label_train_cam2_view1=[label_train_cam2_view1;str2num(subdir_cam2_view1(i).name)];
            train_image_name_cam2_view1=[train_image_name_cam2_view1,image_name];
        end
    end
    for i=3:(length(subdir_cam2_view2))
        fprintf('process test data cam2 view2:%d/%d\n',i,(length(subdir_cam2_view2)));
        image_path_cam2=strcat(param.file_path_cam2,'view2/',subdir_cam2_view2(i).name,'/');
        image_list_cam2=dir(image_path_cam2);
        for j=3:length(image_list_cam2);
            image_name=strcat(image_path_cam2,image_list_cam2(j).name);
            label_train_cam2_view2=[label_train_cam2_view2;str2num(subdir_cam2_view2(i).name)];
            train_image_name_cam2_view2=[train_image_name_cam2_view2,image_name];
        end
    end
    for i=3:(length(subdir_cam2_view4))
        fprintf('process test data cam2 view2:%d/%d\n',i,(length(subdir_cam2_view4)));
        image_path_cam2=strcat(param.file_path_cam2,'view4/',subdir_cam2_view4(i).name,'/');
        image_list_cam2=dir(image_path_cam2);
        for j=3:length(image_list_cam2);
            image_name=strcat(image_path_cam2,image_list_cam2(j).name);
            label_train_cam2_view4=[label_train_cam2_view4;str2num(subdir_cam2_view4(i).name)];
            train_image_name_cam2_view4=[train_image_name_cam2_view4,image_name];
        end
    end
    %generate data for cam3
    for i=3:(length(subdir_cam3_view3))
        fprintf('process test data cam3 view3:%d/%d\n',i,(length(subdir_cam3_view3)));
        image_path_cam3=strcat(param.file_path_cam3,'view3/',subdir_cam3_view3(i).name,'/');
        image_list_cam3=dir(image_path_cam3);
        nameMapLabel_cam3=[nameMapLabel_cam3;str2num(subdir_cam3_view3(i).name)];
        for j=3:length(image_list_cam3);
            image_name=strcat(image_path_cam3,image_list_cam3(j).name);
            label_train_cam3_view3=[label_train_cam3_view3;str2num(subdir_cam3_view3(i).name)];
            train_image_name_cam3_view3=[train_image_name_cam3_view3,image_name];
        end
    end
    for i=3:(length(subdir_cam3_view1))
        fprintf('process test data cam3 view1:%d/%d\n',i,(length(subdir_cam3_view1)));
        image_path_cam3=strcat(param.file_path_cam3,'view1/',subdir_cam3_view1(i).name,'/');
        image_list_cam3=dir(image_path_cam3);
        for j=3:length(image_list_cam3);
            image_name=strcat(image_path_cam3,image_list_cam3(j).name);
            label_train_cam3_view1=[label_train_cam3_view1;str2num(subdir_cam3_view1(i).name)];
            train_image_name_cam3_view1=[train_image_name_cam3_view1,image_name];
        end
    end
    for i=3:(length(subdir_cam3_view2))
        fprintf('process test data cam3 view2:%d/%d\n',i,(length(subdir_cam3_view2)));
        image_path_cam3=strcat(param.file_path_cam3,'view2/',subdir_cam3_view2(i).name,'/');
        image_list_cam3=dir(image_path_cam3);
        for j=3:length(image_list_cam3);
            image_name=strcat(image_path_cam3,image_list_cam3(j).name);
            label_train_cam3_view2=[label_train_cam3_view2;str2num(subdir_cam3_view2(i).name)];
            train_image_name_cam3_view2=[train_image_name_cam3_view2,image_name];
        end
    end
    for i=3:(length(subdir_cam3_view4))
        fprintf('process test data cam3 view4:%d/%d\n',i,(length(subdir_cam3_view4)));
        image_path_cam3=strcat(param.file_path_cam3,'view4/',subdir_cam3_view4(i).name,'/');
        image_list_cam3=dir(image_path_cam3);
        for j=3:length(image_list_cam3);
            image_name=strcat(image_path_cam3,image_list_cam3(j).name);
            label_train_cam3_view4=[label_train_cam3_view4;str2num(subdir_cam3_view4(i).name)];
            train_image_name_cam3_view4=[train_image_name_cam3_view4,image_name];
        end
    end
    for i=1:length(label_train_cam2_view1)
        temp=find(nameMapLabel_cam2==label_train_cam2_view1(i));
        label_train_cam2_view1(i)=temp;
    end
    for i=1:length(label_train_cam2_view2)
        temp=find(nameMapLabel_cam2==label_train_cam2_view2(i));
        label_train_cam2_view2(i)=temp;
    end
    for i=1:length(label_train_cam2_view3)
        temp=find(nameMapLabel_cam2==label_train_cam2_view3(i));
        label_train_cam2_view3(i)=temp;
    end
    for i=1:length(label_train_cam2_view4)
        temp=find(nameMapLabel_cam2==label_train_cam2_view4(i));
        label_train_cam2_view4(i)=temp;
    end
    cam_label_max=size(nameMapLabel_cam2,1);
    for i=1:length(label_train_cam3_view1)
        temp=find(nameMapLabel_cam3==label_train_cam3_view1(i));
        label_train_cam3_view1(i)=temp+cam_label_max;
    end
    for i=1:length(label_train_cam3_view2)
        temp=find(nameMapLabel_cam3==label_train_cam3_view2(i));
        label_train_cam3_view2(i)=temp+cam_label_max;
    end
    for i=1:length(label_train_cam3_view3)
        temp=find(nameMapLabel_cam3==label_train_cam3_view3(i));
        label_train_cam3_view3(i)=temp+cam_label_max;
    end
    for i=1:length(label_train_cam3_view4)
        temp=find(nameMapLabel_cam3==label_train_cam3_view4(i));
        label_train_cam3_view4(i)=temp+cam_label_max;
    end
    save(strcat(param.save_traindata_filename,num2str(split_num_start),'/train_data.mat'),'label_train_cam2_view1','label_train_cam2_view2','label_train_cam2_view3','label_train_cam2_view4', ...
        'label_train_cam3_view1','label_train_cam3_view2','label_train_cam3_view3','label_train_cam3_view4', ...
        'train_image_name_cam2_view1','train_image_name_cam2_view2','train_image_name_cam2_view3','train_image_name_cam2_view4', ...
        'train_image_name_cam3_view1','train_image_name_cam3_view2','train_image_name_cam3_view3','train_image_name_cam3_view4','-v7.3');
    %generate test data
    test_data_cam1=[];
    label_test_cam1=[];
    test_image_name_cam1={};
    test_data_cam2=[];
    label_test_cam2=[];
    test_image_name_cam2={};
    test_data_cam3=[];
    label_test_cam3=[];
    test_image_name_cam3={};
    nameMapLabel=[];
    for i=3:(length(subdir_cam1_view2))
        fprintf('process test data view2:%d/%d\n',i,(length(subdir_cam1_view2)));
        image_path_cam2=strcat(param.file_path_cam1,'view2/',subdir_cam1_view2(i).name,'/');
        image_list_cam2=dir(image_path_cam2);
        nameMapLabel=[nameMapLabel;str2num(subdir_cam1_view2(i).name)];
        for j=3:length(image_list_cam2);
            image_name=strcat(image_path_cam2,image_list_cam2(j).name);
            label_test_cam2=[label_test_cam2;str2num(subdir_cam1_view2(i).name)];
            test_image_name_cam2=[test_image_name_cam2,image_name];
        end
    end
    for i=3:(length(subdir_cam1_view1))
        fprintf('process test data view1:%d/%d\n',i,(length(subdir_cam1_view1)));
        image_path_cam1=strcat(param.file_path_cam1,'view1/',subdir_cam1_view1(i).name,'/');
        image_list_cam1=dir(image_path_cam1);
        for j=3:length(image_list_cam1);
            image_name=strcat(image_path_cam1,image_list_cam1(j).name);
            label_test_cam1=[label_test_cam1;str2num(subdir_cam1_view1(i).name)];
            test_image_name_cam1=[test_image_name_cam1,image_name];
        end
    end
    for i=3:(length(subdir_cam1_view3))
        fprintf('process test data view3:%d/%d\n',i,(length(subdir_cam1_view3)));
        image_path_cam3=strcat(param.file_path_cam1,'view3/',subdir_cam1_view3(i).name,'/');
        image_list_cam3=dir(image_path_cam3);
        for j=3:length(image_list_cam3);
            image_name=strcat(image_path_cam3,image_list_cam3(j).name);
            label_test_cam3=[label_test_cam3;str2num(subdir_cam1_view3(i).name)];
            test_image_name_cam3=[test_image_name_cam3,image_name];
        end
    end
    for i=1:length(label_test_cam1)
        temp=find(nameMapLabel==label_test_cam1(i));
        label_test_cam1(i)=temp;
    end
    for i=1:length(label_test_cam2)
        temp=find(nameMapLabel==label_test_cam2(i));
        label_test_cam2(i)=temp;
    end
    for i=1:length(label_test_cam3)
        temp=find(nameMapLabel==label_test_cam3(i));
        label_test_cam3(i)=temp;
    end
    save(strcat(param.save_testdata_filename,num2str(split_num_start),'/test_data.mat'),'label_test_cam1','label_test_cam2','label_test_cam3','test_image_name_cam1','test_image_name_cam2','test_image_name_cam3','-v7.3');
end