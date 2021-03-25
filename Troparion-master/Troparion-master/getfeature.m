function [PVI , PPF , DPF , PPE] = getfeature(audio_path)
    addpath('IRAPT/IRAPT_web');
    addpath('Perturbation_analysis');
    %以下全部需要转为float32格式
    [s,fs] = audioread(cell2mat(audio_path));
    [Fo, ~, time_marks] = irapt(s, fs, 'irapt1','sustain phonation');  
    [Fo_periods] = WM_phase_const(s,Fo,time_marks,fs);
    
    
    %单位均为%
    PVI    = pathology_vibrato(Fo,time_marks(2),196,8);
    PPF = pitch_petrurbation_factor(Fo_periods,fs);
    DPF = directional_petrurbation_factor(Fo_periods);
    PPE = pitch_period_entropy(Fo);

end

