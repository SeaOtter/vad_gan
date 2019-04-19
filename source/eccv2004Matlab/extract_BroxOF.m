% 
function extract_BroxOF(data_folder, set_name, ext)
    %if nargin < 1
    %%     data_folder = 'g:/hungv/research/WithHung/data/UCSD/UCSDped1';
    %%    data_folder = 'd:/hungv/research/WithHung/data/UCSD/UCSDped1';
    %    data_folder = 'd:/hungv/research/WithHung/data/Avenue_sz240x360fr1';
    %%    data_folder = 'd:/hungv/research/WithHung/data/Avenue/Avenue';
    %%     data_folder = 'd:/hungv/research/WithHung/data/UCSD/UCSDped2';
    %%     set_name = 'train';
    %%     set_name = 'test';
    %   set_name = 'all';
    %%     set_name = 'test8';
    %%     ext = 'tif';
    %    ext = 'png'; % for Avenue
    %end
    H = 240;
    W = 360;

    bdisplay = 0;
%     bdisplay = 1;
    
    list_file = sprintf('%s/%s.lst', data_folder, set_name);
    disp(list_file);
    fid = fopen(list_file, 'r');
    
    video_list = textscan(fid,'%s','delimiter','\n');
    video_list = video_list{1, 1};
    fclose(fid);
    disp(video_list);
    if bdisplay==1
        fig = figure();
    end
    for i = 1:length(video_list)
        video_name = video_list{i};
        disp(sprintf('Video %s\n', video_name));
        image_folder = sprintf('%s/%s', data_folder, video_name);
        image_files = dir(sprintf('%s/*.%s', image_folder, ext));
        image_names = {image_files.name};
        image_names_sorted = sort(image_names);
        num_img = length(image_names_sorted);
        
        
        img_file1 = sprintf('%s/%s', image_folder, image_names_sorted{1});
        im1 = imread(img_file1); 
        im1 = imresize(im1, [H, W]);
%         H = size(im1, 1);
%         W =size(im1, 2); 
        O = zeros([num_img - 1, H, W, 3]);
        for j =2:num_img
            fprintf('[%d]', j);
            img_file2 = sprintf('%s/%s', image_folder, image_names_sorted{j});
            im2 = imread(img_file2); 
            im2 = imresize(im2, [H, W]);
            if length(size(im1))==2
                im1 = repmat(im1, [1, 1, 3]);
            end
            
            if length(size(im2))==2
                im2 = repmat(im2, [1, 1, 3]);
            end
            
            flow = mex_OF(double(im1),double(im2));
            mag = sqrt(flow(:,:,1).^2+flow(:,:,2).^2);
            flow_image = zeros(size(flow, 1),size(flow, 2),3);
            flow_image(:,:,1:2) = flow;
            flow_image(:,:,3) = mag;
            O(j-1, :, :, :) = flow_image;
            if bdisplay == 1
                subplot(2, 3, 1); 
                imshow(im1); 
                
                title(sprintf('%s: Frame %d', video_name, j-1));
                
                subplot(2, 3, 2); 
                imshow(im2); 
                title(sprintf('%s: Frame %d', video_name, j));
                
                subplot(2, 3, 3); 
                colormap('jet');
                imagesc(flow_image(:, :, 1));
                colorbar;
                title('x-axis');
                
                
                subplot(2, 3, 4); 
                colormap('jet');
                imagesc(flow_image(:, :, 2));
                colorbar;
                title('y-axis');
                
                subplot(2, 3, 5); 
                colormap('jet');
                imagesc(flow_image(:, :, 3));
                colorbar;
                title('Magnitude');
                
                subplot(2, 3, 6); 
                colormap('jet');
                imagesc(flow_image);
                colorbar;
                title('Brox Optical Flow');

                pause(0.5);
                
            end
        
%             scale = 16;
%             mag = sqrt(flow(:,:,1).^2+flow(:,:,2).^2)*scale+128;
%             mag = min(mag, 255); 
%             flow = flow*scale+128;
%             flow = min(flow,255);
%             flow = max(flow,0);
% 
%             [x,y,z] = size(flow);
%             flow_image = zeros(x,y,3);
%             flow_image(:,:,1:2) = flow;
%             flow_image(:,:,3) = mag;
% 
%             imwrite(flow_image./255,sprintf('%s/%s/flow_image_%s',save_base,video,frames{k}))
%         
            
            im1 = im2;
            
        end

        O_file = sprintf('%s/feat/%s_sz%dx%d_BroxOF.mat', data_folder, video_name, H, W);
        save(O_file, 'O', '-v7.3');
        disp(sprintf('\nSaved to file %s\n', O_file)); 
    end
    
    
end
