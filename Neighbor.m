function N = Neighbor(S,k)
% select k most similar neighbors in similarity matrix S and normalized

[m,n] = size(S);
S = S - diag(diag(S));
N = zeros(m,n);
for i=1:m
   [row, index] = sort(S(i,:),'descend');
   neig = row(1:k);
   pos = index(1:k);
   
   for num = 1: k
       N(i,pos(num)) = neig(num)/sum(neig);
   end
end