function [  ] = audio_mat_to_txt(  )
    mat_loc = 'E:\DR\AudioFeatures\FeaturesandFormants\FeaturesALL';
    %location for the new features
    csv_loc = 'E:\DR\NewAudio';

    cd (mat_loc);
    files = dir;
    file_names = {files(~[files.isdir]).name};
    for k=1:length(file_names)
        disp(file_names{k});
        load (file_names{k});
        %remove vuv feature not equal to 1
        vuv = features(:,2) ~= 1;
        new_feat = features;
        new_feat(vuv,:) = [];
        %remove creaky feature(vuv)
        new_feat = new_feat(:,any(diff(new_feat,1)));
        csvwrite(strcat(csv_loc,'\',file_names{k}(1:end-4),'.txt'),new_feat);
    end
end

