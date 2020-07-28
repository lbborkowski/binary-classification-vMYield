clear all
close all
clc

% define plate with hole geometry and mesh
nh=16; % 2*nh is the number of elements along an edge
L=1; % length of the plate
rad=0.1; % radius of the hole

% load the variable containing the nodal locations and element connectivity
load('m.mat');

% alternatively, you can download the hole_mesh.m function from
% http://compmech.lab.asu.edu/data/hole_mesh.m 
% and uncomment the following line
%m = hole_mesh(nh, L, rad);

p.vertices = m.x'; % nodal locations
p.faces = m.conn'; % element connectivity
p.facecolor = 'w'; % element face color

% plot mesh
patch(p)
axis square
axis off

% plot nodal positions
figure('color','w')
plot(m.x(1,:),m.x(2,:),'ok')
axis([min(m.x(1,:)) max(m.x(1,:)) min(m.x(2,:)) max(m.x(2,:))])
axis square
axis off

% convert nodal coordinates from cartesian to cylindrical coordinate system
[theta,rho]=cart2pol(m.x(1,:),m.x(2,:));

sig=1; % applied stress in x direction
a=rad; % radius of hole
r=rho; % rho

% calculate 2D stress components in cylindrical coordinate system
sig_rr=sig/2*((1-a^2./r.^2)+(1+3*a^4./r.^4-4*a^2./r.^2).*cos(2*theta));
sig_tt=sig/2*((1+a^2./r.^2)-(1+3*a^4./r.^4).*cos(2*theta));
sig_rt=-sig/2*((1-3*a^4./r.^4+2*a^2./r.^2).*sin(2*theta));

% compute bounds for each of the three stress components
range_sig_rr=[min(sig_rr) max(sig_rr)]
range_sig_tt=[min(sig_tt) max(sig_tt)]
range_sig_rt=[min(sig_rt) max(sig_rt)]

% plot sig_rr
figure('color','w')
colormap jet
scatter(m.x(1,:),m.x(2,:),30,sig_rr,'filled')
axis([min(m.x(1,:)) max(m.x(1,:)) min(m.x(2,:)) max(m.x(2,:))])
axis off
title('\sigma_{rr}','fontsize',24);
cb=colorbar('fontsize',24);

% plot sig_tt
figure('color','w')
colormap jet
scatter(m.x(1,:),m.x(2,:),30,sig_tt,'filled')
axis([min(m.x(1,:)) max(m.x(1,:)) min(m.x(2,:)) max(m.x(2,:))])
axis off
title('\sigma_{\theta\theta}','fontsize',24);
cb=colorbar('fontsize',24);

% plot sig_rt
figure('color','w')
colormap jet
scatter(m.x(1,:),m.x(2,:),30,sig_rt,'filled')
axis([min(m.x(1,:)) max(m.x(1,:)) min(m.x(2,:)) max(m.x(2,:))])
axis off
title('\sigma_{r\theta}','fontsize',24);
cb=colorbar('fontsize',24);

% calculate von Mises stress as a function of
% 2D stress components in cylindrical coordinates
sig_vM=sqrt(sig_rr.^2-sig_rr.*sig_tt+sig_tt.^2+3*sig_rt.^2);

% plot von Mises stress
figure('color','w')
colormap jet
scatter(m.x(1,:),m.x(2,:),30,sig_vM,'filled')
axis([min(m.x(1,:)) max(m.x(1,:)) min(m.x(2,:)) max(m.x(2,:))])
axis off
title('\sigma_{vM}','fontsize',24);
cb=colorbar('fontsize',24);

% calculate yield
yield=zeros(size(sig_vM));
yield(sig_vM>=1.15)=1;

% plot yield
figure('color','w')
colormap jet
scatter(m.x(1,:),m.x(2,:),30,yield,'filled')
axis([min(m.x(1,:)) max(m.x(1,:)) min(m.x(2,:)) max(m.x(2,:))])
axis off
cb=colorbar('fontsize',24);

% print node locations and stress values to files
% these are read by Jupyter notebook
dlmwrite('nodes.txt',m.x') 
sigMat=[sig_rr' sig_tt' sig_rt'];
dlmwrite('stress.txt',sigMat)
dlmwrite('yield.txt',yield)




