% Transform (Shear and Occlude data)
function [sheared] = Shear(input, a)
    tform = affine2d([1 0 0; a 1 0; 0 0 1]);
    %sheared = zeros(784, 10000);     
    dims = size(input)
    sheared = zeros(dims(1), dims(2)); 
    for i = 1:10000 
        A = (reshape(input(:, i), 28, 28)); 
        % Shear the image
        B = imwarp(A, tform);
        width = size(B, 2); 
        height = size(B, 1); 
        
        % Just a guess
        low = floor(.25 * width); 
        % crop the image
        B = imcrop(B,[low, 1, height - 1, height]);
        sheared(:, i) = reshape(B, size(1), 1);
    end
end 