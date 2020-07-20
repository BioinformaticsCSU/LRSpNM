function Y=preprocess_KNNC(Y,Sd,St,K)
    
    N_Sd = Neighbor(Sd,K);
    N_St = Neighbor(St,K);
    % average computed values of the modified 0's from the drug and target
    % sides while preserving the 1's that were already in Y 
    Y2 = 0.5*N_Sd*Y+0.5*Y*N_St';
    Y = max(Y,Y2);
end