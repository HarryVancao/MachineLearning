X = 255*reshape(test_images, 28, 28, 10000); 
fp = fopen("shear-idx3-ubyte", "wb"); 
fwrite(fp,2051,'integer*4', 'ieee-be');
fwrite(fp,10000,'integer*4', 'ieee-be');
fwrite(fp,28,'integer*4', 'ieee-be');
fwrite(fp,28,'integer*4', 'ieee-be');


for i = 1:10000
    for j = 1:28
        for k = 1:28
            fwrite(fp,X(j, k, i),'uchar');
        end
    end 
end 

fclose(fp);