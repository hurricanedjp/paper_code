fid=fopen('result.txt','at+');
fprintf(fid,' µ—È–Ú¥Œ\tmse\t\terror\t\ttime\n');
for i=1:15
    t1=clock;
    Copy_of_cnnexamples;
    t2=clock;
    fprintf(fid,'   %g\t %5.4f\t  %5.4f%%\t %g\n',i,cnn.result(end),er*100,etime(t2,t1));
end
fclose(fid);