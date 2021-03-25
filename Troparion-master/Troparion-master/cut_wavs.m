function cut_wavs(file_dir,output_dir,t,t_overlap)
        %输入有四个，源目录，输出目录，截取时长，不重叠时长
        %解释一下不重叠时长，1个55秒的音乐文件，设置截取时长30s，不重叠时长为30s，就只能生成一
        %段，0-30s；设置不重叠是15，生成2段，0-30s，15-30s，以此类推；
files=dir(file_dir);
for i =3:length(files)          % 解释一下，从3开始是因为前两个是父目录".."和当前目录"."
    if ~exist(output_dir)       % 判断保存目录是否存在
        mkdir(output_dir);      % 若不存在，在当前目录中产生一个子目录‘Figure’
    end 
    if files(i).isdir           % 判断当前文件名对用的是否是文件夹
        temp_dir=file_dir;      % 是，开始递归，遍历所有
        now_dir=strcat(temp_dir,'/',files(i).name);
        now_output_dir=strcat(output_dir,'/',files(i).name);
        cut_wavs(now_dir,now_output_dir,t,t_overlap);
        clear temp_dir;
    elseif files(i).name(end-2:end)=='wav' % 不是文件夹，判断是否是wav文件
        wavfile_name_new = strcat(file_dir,'/', files(i).name);    
        [y,fs]=audioread(wavfile_name_new);% 读取音乐数值与参数，fs是采样率     
        last_time=length(y)/fs;            % 音乐持续时间长度
        num=fix((last_time-t)/t_overlap+1);% 切割后音乐的份数
        for k=1:num                        % 对每首音乐进行切割并命名    
            starttime=(k-1)*t_overlap;
            endtime=(k-1)*t_overlap+t;
            y_out=y(starttime*fs+1:endtime*fs+1,:);
            % 这里解释一下，我的wav文件是双声道的所以数组是nx2的结构，如果你是单声道
            % 那么应该是nx1，需要把后面的",:"去掉     
            filename=strcat(output_dir,'/',num2str(i-2),'-',num2str(k),'.wav');
            attentions=strcat('Saving...',filename) 
            audiowrite(filename,y_out,fs);
        end     
    end
    end
end

