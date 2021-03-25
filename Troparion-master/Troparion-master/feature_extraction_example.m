close all;

addpath('IRAPT/IRAPT_web');
addpath('Perturbation_analysis');

PathRoot = 'D:\data\PKS\NeuroSpeech-master\src\wavtest\';

for i = 1:4
    Path = strcat(PathRoot,num2str(i),'\');                   
    File = dir(fullfile(Path,'*.wav')); 
    FileNames = {File.name}';

    Length_Names = size(FileNames,1);   
    for k = 1 : Length_Names
        K_Trace = strcat(Path, FileNames(k));
         disp(K_Trace);
        [s,fs] = audioread(cell2mat(K_Trace));

    [Fo, ~, time_marks] = irapt(s, fs, 'irapt1','sustain phonation');  

    % Segmentation of signal onto fundamental periods
    [Fo_periods] = WM_phase_const(s,Fo,time_marks,fs);
%     [periods_Amp]= amp_extract(Fo_periods,s);

    %% Jitter
    J_loc  = shim_local(Fo_periods);
    J_rap  = jitter_rap(Fo_periods);
    J_ppq5 = jitter_ppq5(Fo_periods);
    J_apq55 = shimmer_apq(Fo_periods,55);

%     S_loc   = shim_local(periods_Amp);
%     S_apq3  = shim_apq3(periods_Amp);
%     S_apq5  = shim_apq5(periods_Amp);
%     S_apq11 = shim_apq11(periods_Amp);
%     S_apq55  = shimmer_apq(periods_Amp,55);

    PVI    = pathology_vibrato(Fo,time_marks(2),196,8);

    PPF = pitch_petrurbation_factor(Fo_periods,fs);
    DPF = directional_petrurbation_factor(Fo_periods);
    PPE = pitch_period_entropy(Fo);

    % fprintf('Jitter:local  = %1.2f %% \n', J_loc);
    % fprintf('Jitter:RAP    = %1.2f %% \n', J_rap);
    % fprintf('Jitter:PPQ5   = %1.2f %%\n', J_ppq5);            
    % fprintf('Jitter:PPQ55  = %1.2f %%\n', J_apq55);
    % fprintf('Shimmer:local = %1.2f %% \n', S_loc);
    % fprintf('Shimmer:APQ3  = %1.2f %% \n', S_apq3);
    % fprintf('Shimmer:APQ5  = %1.2f %% \n', S_apq5);
    % fprintf('Shimmer:APQ11 = %1.2f %% \n', S_apq11);          
    % fprintf('Shimmer:APQ55 = %1.2f %% \n', S_apq55);
    % fprintf('PVI           = %1.3f \n', PVI);
    % fprintf('PPF           = %1.3f \n', PPF);
    % fprintf('DPF           = %1.2f %%\n', DPF);
    % fprintf('PPE           = %1.2f \n', PPE);


    A = [PVI  PPF  DPF  PPE];
    disp(A)
%     dlmwrite('test.txt',A,'newline','pc');
    fid=fopen('Troparion.txt','A');
    fprintf(fid,'%s,%f,%f,%f,%f\n',cell2mat(FileNames(k)),A);
    fclose(fid);
        
    end
end


