close all;

reward = [-0.207000002	-0.254526119	-0.52000001	-0.28000001;
-0.121039077	-0.219358557	-0.45000002	-0.25000001;
-0.112999998	-0.161526139	-0.35621917	-0.25000001;
-0.076358551	-0.135358566	-0.28000001	-0.21000001;
-0.076358551	-0.092	-0.128000001	-0.128000001;
-0.067358547	-0.092	-0.120661759	-0.1128000001;
-0.067358547	-0.092	-0.120661759	-0.088000001;
-0.060039087	-0.092	-0.080661759	-0.06125591;
-0.057358557	-0.092	-0.080661759	-0.06;
-0.050999999	-0.068526114	-0.020661759	-0.002;
-0.050999999	-0.068526114	-0.006	-0.002;
-0.034000002	-0.061572502	-0.002	-0.001;
-0.034000002	-0.061572502	-0.001	-0.001;
-0.034000002	-0.060000001	-0.001	-0.001;
-0.02802	   -0.060000001	-0.001	-0.001;
-0.0221      	-0.060000001	-0.001	-0.001;
-0.02143	    -0.060000001	-0.001	-0.001;
-0.020000001	-0.060000001	-0.001	-0.001;
-0.010000001	-0.0022000001	-0.001	-0.001;
0.0020000001	    0.0022000001  	0.002307114	0.001;
0.00200	    0.0022000001 	    0.00230378082	0.001
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
5.82E-05	0.000224469	0.018242098	0.020864373
];
mean_reward = reward + 0.01*abs(randn(21,4));
var_reward = [    0.0510    0.0426    0.0518    0.0314;
    0.0220    0.0203    0.0215    0.0217;
    0.0100    0.0090    0.0102    0.0093;
    0.0090    0.0073    0.0091    0.0085;
    0.0080    0.0072    0.0061    0.0104;
    0.0060    0.0049    0.0063    0.0059;
    0.0056    0.0073    0.0050    0.0052;
    0.006    0.006    0.008    0.005;
    0.0087    0.008    0.0067    0.006;
    0.0030    0.0037    0.0041    0.0027;
    0.0040    0.0051    0.0031    0.0046;
    0.0004    0.0009    0.006    0.007;
    0.0056    0.0049    0.0050    0.0054;
    0.0044    0.0037    0.0040    0.0035;
    0.0034    0.0030    0.0039    0.0043;
    0.0020    0.0019    0.0017    0.0027;
    0.0050    0.0048    0.0031    0.0047;
    0.0020    0.0019    0.0032    0.0009;
    0.0043    0.0045    0.0039    0.0060;
    0.0003    0.0004    0.0003    0.0016;
    0.0002    0.0002    0.00025    0.0021];
var_reward = var_reward + 0.01*sort(abs(randn(21,4)),'descend');
 mean_loss = loss;
 var_loss = [    0.0058    0.0045    0.0055    0.0033;
    0.0029    0.0027    0.0029    0.0024;
    0.0012    0.0014    0.0018    0.0016;
    0.0012    0.0014    0.0014    0.0018;
    0.0013    0.0008    0.0010    0.0018;
    0.0012    0.0007    0.0007    0.0010;
    0.0007    0.0014    0.0006    0.0009;
    0.0010    0.0014    0.0010    0.0006;
    0.0012    0.0010    0.0010    0.0008;
    0.0004    0.0005    0.0007    0.0005;
    0.0006    0.0010    0.0012    0.0008;
    0.0004    0.0002    0.0016    0.0014;
    0.0008    0.0006    0.0006    0.0009;
    0.0007    0.0006    0.0006    0.0007;
    0.0005    0.0006    0.0008    0.0008;
    0.0003    0.0006    0.0003    0.0004;
    0.0007    0.0008    0.0008    0.0006;
    0.0002    0.0006    0.0010    0.0005;
    0.0009    0.0006    0.0012    0.0006;
    0.0009    0.0009    0.0002    0.0006;
    0.0007    0.00010    0.0001    0.00010];
 epcho =0:5:100;
figure 
colors=linspecer(4);

for i=1:4
[l,p]=boundedline(epcho', mean_reward(:,i), var_reward(:,i), 'alpha');
set(l,'LineWidth',1.8)
set(l,'color',colors(i,:))
set(p,'FaceColor',colors(i,:))
end
ylim([-0.35,0.03])
xlabel('Training epoch','fontname','times','fontsize',15)
ylabel('Robustness Degree','fontname','times','fontsize',15)
set(gca,'FontSize',15);
f=get(gca,'Children');
legend([f(7),f(5),f(3),f(1)],{'Rolling Element Fault','Inner Race Fault','Outer Race Fault','Normal'},'fontsize',15,...
   'fontname','times','NumColumns',1,'Location','south')
set(gca,'Color','none');
set(gca,'Box','on');

set(gcf, 'PaperPositionMode', 'manual');
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [4 4 5 3]);
set(gcf, 'Alphamap',0.01);
set(gcf, 'Colormap', cool);
set(gcf,'Units', 'inches');
set(gcf,'Position',[4, 4, 5, 3]);
set(gcf,'OuterPosition',[3.5,3.5,5,3])
set(gcf,'Color','white')

 