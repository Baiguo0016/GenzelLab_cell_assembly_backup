fileData = load("RGS14_GMM.mat");
nr_inter_neuron = length(fileData.RGS14_GMM(1).Interneurons);

if nr_inter_neuron ~= 0
    inter_neurons = string([nr_inter_neuron 1]);
    for i = 1:nr_inter_neuron
        inter_neuron = fileData.RGS14_GMM.Interneurons(i).WFM_Titles;
        inter_neurons(i) = inter_neuron;
    end

    A = inter_neurons;
    fid = fopen('inter_neurons_RGS14.txt','w');
    Anew = [A repmat(newline,size(A,1),1)];
    fprintf(fid,'%s,',Anew.');
    fclose(fid);
else   
    fid = fopen('inter_neurons_RGS14.txt','w');
    fclose(fid);
end