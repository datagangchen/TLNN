close all
reward = [-0.207000002	-0.254526119	-0.052000001	-0.028000001;
-0.121039077	-0.219358557	-0.045000002	-0.028000001;
-0.112999998	-0.161526139	-0.035621917	-0.028000001;
-0.076358551	-0.135358566	-0.028000001	-0.028000001;
-0.076358551	-0.092	-0.028000001	-0.028000001;
-0.067358547	-0.092	-0.020661759	-0.028000001;
-0.067358547	-0.092	-0.020661759	-0.028000001;
-0.060039087	-0.092	-0.020661759	-0.006125591;
-0.057358557	-0.092	-0.020661759	-0.006;
-0.050999999	-0.068526114	-0.020661759	-0.002;
-0.050999999	-0.068526114	-0.006	-0.002;
-0.034000002	-0.061572502	-0.002	-0.001;
-0.034000002	-0.061572502	-0.001	-0.001;
-0.034000002	-0.060000001	-0.001	-0.001;
-0.02802	-0.060000001	-0.001	-0.001;
-0.0221	-0.060000001	-0.001	-0.001;
-0.02143	-0.060000001	-0.001	-0.001;
-0.020000001	-0.060000001	-0.001	-0.001;
-0.020000001	-0.052000001	-0.001	-0.001;
-0.020000001	-0.052000001	0.002307114	0.001;
0.01643	0.01121	0.015378082	0.00123
];
loss = [0.049804863	0.048044033	0.049358923	0.054547053;
0.038007345	0.023343801	0.04474763	0.04889308;
0.034583285	0.019311117	0.04144	0.051120542;
0.02733505	0.012166636	0.03915332	0.04581329;
0.02202991	0.008872253	0.041968506	0.044501137;
0.017742882	0.00851371	0.03463875	0.041124146;
0.012466381	0.006274578	0.035245344	0.04551895;
0.008988192	0.005865696	0.031409044	0.03724078;
0.005222965	0.003956608	0.03297847	0.037872575;
0.003193955	0.004184822	0.029184904	0.03752845;
0.002724382	0.003472381	0.027930679	0.038376804;
0.001267931	0.002814075	0.028390585	0.03382783;
0.000624265	0.002360223	0.02526376	0.032760307;
0.000360611	0.002326549	0.022799378	0.032351423;
0.000338061	0.001735531	0.0235934	0.030328868;
0.000135894	0.001691125	0.02147603	0.02915228;
9.01E-05	0.001178579	0.020053988	0.028051836;
0.000268264	0.000881605	0.021117285	0.02786744;
4.25E-05	0.000213229	0.017952187	0.024903057;
5.34E-05	0.000961786	0.019066967	0.025628975;
2.82E-05	0.000224469	0.018242098	0.020864373
];

epoch = [0:5:100];
figure

subplot(1,2,1)
plot(epoch, reward(:,1),'-g',epoch,reward(:,2),'-.r',epoch,reward(:,3),':b','linewidth',1.5)
xlabel('Training epoch','fontname','times','fontsize',15)
ylabel('Optimal robustness degree','fontname','times','fontsize',15)
set(gca,'FontSize',15);
legend({'Rolling Element Fault','Inner Race Fault','Outer Race Fault'},'fontsize',15,...
    'fontname','times','NumColumns',1,'Location','south')
set(gca,'Color','none');
set(gca,'Box','on');
subplot(1,2,2)
plot(epoch, loss(:,1),'-g',epoch,loss(:,2),'-.r',epoch,loss(:,3),':b','linewidth',1.5)
xlabel('Training epoch','fontname','times','fontsize',15)
ylabel('Regression loss','fontname','times','fontsize',15)

legend({'Rolling Element Fault','Inner Race Fault','Outer Race Fault'},'fontsize',15,'fontname','times','NumColumns',1)

set(gca,'FontName','times')
set(gca, 'FontSize',15)
 set(gca,'Color','none');
set(gca,'Box','on');

set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [4 4 8 4]);
set(gcf, 'Alphamap',0.01);
set(gcf, 'Colormap', cool);
set(gcf,'Units', 'inches');
set(gcf,'Position',[4, 4, 8, 4]);
set(gcf,'OuterPosition',[3.5,3.5,8.5,5])
set(gcf,'Color','white')

 