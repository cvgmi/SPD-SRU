n = 3;
a = 0.5;

A = randn(n);
A = (A + A')/2+pascal(n);
[V,D]=eig(A);
D(D<=0.2)=0.2;
A = V*D*V';

B = randn(n);
B = (B + B')/2+pascal(n);
[V,D]=eig(B);
D(D<=0.2)=0.2;
B = V*D*V';

r = FM(A,B,a);
[~,D]=eig(r);
disp(r);


function result = FM(A,B,a)
AB = inv(A)*B;
% AB = A^-0.5 * B * A^-0.5;
temp = AB+(2*a-1)^2/4*(eye(size(A,1))-AB)^2-(2*a-1)/2*(eye(size(A,1))-AB);
[V,D]=eig(temp);
D = sqrt(D);
temp = V*D*inv(V);
% temp = sqrtm(temp);
result = A*temp;

end