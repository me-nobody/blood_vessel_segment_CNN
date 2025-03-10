# load libraries
library(tidyverse)
library(ggplot2)
library(GGally)

hist(area_cd31$area,breaks=100)
hist(area_claudin$area,breaks =100)
hist(area_pdgfrb$area,breaks = 100)
# set an upper limit
area_cd31_limit <- area_cd31[area_cd31$area<=4000,]
area_claudin_limit <- area_claudin[area_claudin$area<=4000,]
area_pdgfrb_limit <- area_pdgfrb[area_pdgfrb$area<=4000,]
# replot
hist(area_cd31_limit$area,breaks=200)
hist(area_claudin_limit$area,breaks =200)
hist(area_pdgfrb_limit$area,breaks = 200)
# seems claudin has NA values
sum(is.na(area_claudin_limit$area)) # improper subsetting
# setup breakpoints
breakpoints = c(seq(0,3000,100),Inf)
# bin the dataset based on breakpoints
categories <-cut(area_cd31$area,breakpoints)
# add the categories to all the datasets
area_cd31$category <-categories
area_claudin$category <- cut(area_claudin$area,breakpoints)
area_pdgfrb$category <- cut(area_pdgfrb$area,breakpoints)
area_laminin$category <- cut(area_laminin$area,breakpoints)
# now we have to obtain counts based on the bins
# first separate the tumor and infil datasets
area_cd31_infil <- area_cd31[area_cd31$region=="infil",]
area_cd31_tumor <- area_cd31[area_cd31$region=="tumor",]
area_claudin_infil <- area_claudin[area_claudin$region=="infil",]
area_claudin_tumor <- area_claudin[area_claudin$region=="tumor",]
area_pdgfrb_infil <- area_pdgfrb[area_pdgfrb$region=="infil",]
area_pdgfrb_tumor <- area_pdgfrb[area_pdgfrb$region=="tumor",]
area_laminin_infil <- area_laminin[area_laminin$region=="infil",]
area_laminin_tumor<- area_laminin[area_laminin$region=="tumor",]


# now groupby the categories
area_cd13_infil_counts<- area_cd31_infil%>%group_by(category)%>%summarise(count=n())
area_cd31_tumor_counts<- area_cd31_tumor%>%group_by(category)%>%summarise(count=n())
area_claudin_infil_counts<- area_claudin_infil%>%group_by(category)%>%summarise(count=n())
area_claudin_tumor_counts<- area_claudin_tumor%>%group_by(category)%>%summarise(count=n())
area_pdgfrb_infil_counts<- area_pdgfrb_infil%>%group_by(category)%>%summarise(count=n())
area_pdgfrb_tumor_counts<- area_pdgfrb_tumor%>%group_by(category)%>%summarise(count=n())

area_laminin_infil_counts<- area_laminin_infil%>%group_by(category)%>%summarise(count=n())
area_laminin_tumor_counts<- area_laminin_tumor%>%group_by(category)%>%summarise(count=n())

# after binning some datasets have some categories missing; filling the datasets
setdiff(area_cd31_infil_counts$category,area_claudin_tumor_counts$category)
area_claudin_tumor_counts<-rbind(area_claudin_tumor_counts,c("(2.8e+03,2.9e+03]",0))

setdiff(area_cd31_infil_counts$category,area_pdgfrb_infil_counts$category)
area_pdgfrb_infil_counts<-rbind(area_pdgfrb_infil_counts,c("(2.5e+03,2.6e+03]",0))
area_pdgfrb_infil_counts<-rbind(area_pdgfrb_infil_counts,c("(2.6e+03,2.7e+03]",0))
area_pdgfrb_infil_counts<-rbind(area_pdgfrb_infil_counts,c("(2.7e+03,2.8e+03]",0))
area_pdgfrb_infil_counts<-rbind(area_pdgfrb_infil_counts,c("(2.8e+03,2.9e+03]",0))


setdiff(area_cd31_infil_counts$category,area_pdgfrb_tumor_counts$category)
area_pdgfrb_tumor_counts<-rbind(area_pdgfrb_tumor_counts,c("(2.7e+03,2.8e+03]",0))
area_pdgfrb_tumor_counts<-rbind(area_pdgfrb_tumor_counts,c("(2.8e+03,2.9e+03]",0))
# convert counts to fractions
area_cd31_infil_counts<-area_cd31_infil_counts%>%mutate(fraction = count/sum(count))
area_claudin_infil_counts<-area_claudin_infil_counts%>%mutate(fraction = count/sum(count))
area_pdgfrb_infil_counts<-area_pdgfrb_infil_counts%>%mutate(fraction = count/sum(count))
area_laminin_infil_counts<-area_laminin_infil_counts%>%mutate(fraction = count/sum(count))


area_cd31_tumor_counts<-area_cd31_tumor_counts%>%mutate(fraction = count/sum(count))
area_claudin_tumor_counts<-area_claudin_tumor_counts%>%mutate(fraction = count/sum(count))
area_pdgfrb_tumor_counts<-area_pdgfrb_tumor_counts%>%mutate(fraction = count/sum(count))
area_laminin_tumor_counts<-area_laminin_tumor_counts%>%mutate(fraction = count/sum(count))

# rename the categories
area_category <- c("100","200","300","400","500","600","700","800","900","1000","1100","1200","1300","1400","1500","1600","1700","1800","1900","2000","2100","2200","2300","2400","2500","2600","2700","2800","2900","3000",">3000")
area_category <- as.factor(area_category)
area_cd31_infil_counts$category <-area_category
area_cd31_tumor_counts$category <-area_category
area_claudin_infil_counts$category <-area_category
area_claudin_tumor_counts$category <-area_category
area_pdgfrb_infil_counts$category <-area_category
area_pdgfrb_tumor_counts$category <-area_category

area_laminin_infil_counts$category <-area_category
area_laminin_tumor_counts$category <-area_category
# scatterplot
plot(log(area_cd31_infil_counts$fraction),log(area_claudin_infil_counts$fraction),col="red")
points(log(area_cd31_tumor_counts$fraction),log(area_claudin_tumor_counts$fraction),col="blue")
# density plot
ggplot(area_cd31,aes(x=log(area),fill=region))+ geom_density(alpha=0.4)+ggtitle("CD31")
ggplot(area_claudin,aes(x=log(area),color=region))+ geom_density()+ggtitle("Claudin")
ggplot(area_laminin,aes(x=log(area),color=region))+ geom_density()+ggtitle("Laminin")
ggplot(area_pdgfrb,aes(x=log(area),color=region))+ geom_density()+ggtitle("PDGFRb")
# formatted plot
# CD31
ggplot(data=area_cd31,aes(x=sqrt(area),fill=region)) + geom_density(alpha=0.4)+
  theme(plot.title = element_text(color="black", size=22, face="bold.italic"),
                      axis.title.x = element_text(color="black", size=22, face="bold"),
                      axis.title.y = element_text(color="black", size=22, face="bold"))+
  theme(axis.text = element_text(color="black",size=20,face="bold",angle=0))+
  ggtitle("distribution of area of CD31") + theme(panel.background = element_blank())+
      theme(legend.title = element_text(color = "black",size = 18),
      legend.text = element_text(color = "black",size=16))+xlim(-10,100)
# Claudin
ggplot(data=area_claudin,aes(x=log(area),fill=region)) + geom_density(alpha=0.4)+
  theme(plot.title = element_text(color="black", size=22, face="bold.italic"),
        axis.title.x = element_text(color="black", size=22, face="bold"),
        axis.title.y = element_text(color="black", size=22, face="bold"))+
  theme(axis.text = element_text(color="black",size=20,face="bold",angle=0))+
  ggtitle("distribution of area of Claudin") + theme(panel.background = element_blank())
# Laminin
ggplot(data=area_laminin,aes(x=sqrt(area),fill=region)) + geom_density(alpha=0.4)+
  theme(plot.title = element_text(color="black", size=22, face="bold.italic"),
        axis.title.x = element_text(color="black", size=22, face="bold"),
        axis.title.y = element_text(color="black", size=22, face="bold"))+
  theme(axis.text = element_text(color="black",size=20,face="bold",angle=0))+
  ggtitle("distribution of area of Laminin") + theme(panel.background = element_blank())+
  theme(legend.title = element_text(color = "black",size = 18),
        legend.text = element_text(color = "black",size=16))+xlim(-10,100)

# PDGFRb
ggplot(data=area_pdgfrb,aes(x=log(area),fill=region)) + geom_density(alpha=0.4)+
  theme(plot.title = element_text(color="black", size=22, face="bold.italic"),
        axis.title.x = element_text(color="black", size=22, face="bold"),
        axis.title.y = element_text(color="black", size=22, face="bold"))+
  theme(axis.text = element_text(color="black",size=20,face="bold",angle=0))+
  ggtitle("distribution of area of PDGFRb") + theme(panel.background = element_blank())
# create a metadata set
area_cd31_infil_counts$region<-"infil"
area_cd31_tumor_counts$region<-"tumor"
area_claudin_infil_counts$region<-"infil"
area_claudin_tumor_counts$region<-"tumor"
area_laminin_infil_counts$region<-"infil"
area_laminin_tumor_counts$region<-"tumor"
area_pdgfrb_infil_counts$region<-"infil"
area_pdgfrb_tumor_counts$region<-"tumor"


area_cd31_infil_counts$marker<-"CD31"
area_cd31_tumor_counts$marker<-"CD31"
area_claudin_infil_counts$marker<-"Claudin"
area_claudin_tumor_counts$marker<-"Claudin"
area_laminin_infil_counts$marker<-"Laminin"
area_laminin_tumor_counts$marker<-"Laminin"
area_pdgfrb_infil_counts$marker<-"PDGFRb"
area_pdgfrb_tumor_counts$marker<-"PDGFRb"

area_markers= rbind(area_cd31_infil_counts,area_cd31_tumor_counts,
                    area_claudin_infil_counts,area_claudin_tumor_counts,
                    area_laminin_infil_counts,area_laminin_tumor_counts,
                    area_pdgfrb_infil_counts,area_pdgfrb_tumor_counts)

area_markers$region<-as.factor(area_markers$region)
area_markers$marker<-as.factor(area_markers$marker)

infil_fractions<-cbind(area_cd31_infil_counts$fraction,
      area_claudin_infil_counts$fraction,
      area_laminin_infil_counts$fraction,
      area_pdgfrb_infil_counts$fraction)
infil_fractions <- as.data.frame(infil_fractions)
names(infil_fractions) <- c("CD31","Claudin","Laminin","PDGFRb")
# log of fraction
infil_area_percentage<-infil_fractions*100
#infil_area_fractions<-infil_fractions+0.001
apply(infil_area_fractions,2,log)
# plot the scatterplot
ggpairs(infil_fractions)
ggpairs(infil_area_fractions)
# tumor
tumor_fractions<-cbind(area_cd31_tumor_counts$fraction,
                       area_claudin_tumor_counts$fraction,
                       area_laminin_tumor_counts$fraction,
                       area_pdgfrb_tumor_counts$fraction)
tumor_fractions <- as.data.frame(tumor_fractions)
names(tumor_fractions) <- c("CD31","Claudin","Laminin","PDGFRb")
tumor_area_percentage<-tumor_fractions*100

# plot the scatterplot
ggpairs(tumor_fractions)


# custom scaling in ggpairs
limitRange <- function(data, mapping, ...) { 
  ggplot(data = data, mapping = mapping, ...) + 
    geom_point(...) + 
    geom_smooth(method = "lm", se = FALSE) +
    ylim(0,60)+xlim(0,60)
}

# This is how you specify which part of the image will be
# plotted using your function.
ggpairs(infil_area_percentage, aes(alpha=0.4),lower = list(continuous = limitRange))+ggtitle("infiltration zone")
# create a label for small and large vessels
size_labels<-as.factor(c(rep(c("small"),10),rep(c("large"),21)))
infil_area_percentage$size<-size_labels
tumor_area_percentage$size<-size_labels
ggpairs(tumor_area_percentage, aes(alpha=0.4),lower = list(continuous = limitRange))+ggtitle("tumor zone")
# setting >3000 to 4000 to convert column to numeric
area_markers$category<-as.character(area_markers$category)
area_markers$category[is.na(area_markers$category)]<-4000
area_markers$category<- as.integer(area_markers$category)
# check specific to claudin
area_claudin <-area_markers%>%filter(marker=='Claudin')
area_pdgfrb <-area_markers%>%filter(marker=='PDGFRb')
ggplot(area_claudin,aes(x=category,y=fraction,color=region,fill=region))+geom_col()+theme(axis.text = element_text(color="black",size=10,face="bold",angle=90))
ggplot(area_pdgfrb,aes(x=category,y=fraction,color=region))+geom_point()+theme(axis.text = element_text(color="black",size=10,face="bold",angle=90))
# ggcol better depicts the data
area_gg<-ggplot(area_markers,aes(x=category,y=fraction,color=region,fill=region))+
  geom_col(alpha=0.4)+  theme(plot.title = element_text(color="black", size=16, face="bold.italic"),
        axis.title.x = element_text(color="black", size=16, face="bold"),
        axis.title.y = element_text(color="black", size=16, face="bold"))+
  theme(axis.text = element_text(color="black",size=6,face="bold",angle=90))+labs(x="size stained area",y="fraction of total stained area")
  ggtitle("distribution of area of markers") + theme(panel.background = element_blank())
area_gg + facet_grid(region~marker)+theme(strip.text.x = element_text(size=12, color="black",face="bold.italic"))+
                                  theme(strip.text.y = element_text(size=12, color="black", face="bold.italic"))+
                                                                 theme(panel.background = element_blank())
# let us get the really large blood vessels
large_area<-area_markers%>%filter(category>2500)
small_area<-area_markers%>%filter(category>100 & category<2500)

markers_large<-large_area%>%group_by(marker,region)%>%summarise(percentage=sum(fraction))
markers_large<-markers_large%>%mutate(percentage=100*percentage)

markers_small<-small_area%>%group_by(marker,region)%>%summarise(percentage=sum(fraction))
markers_small<-markers_small%>%mutate(percentage=100*percentage)

ggplot(markers_large,aes(x=marker,y=percentage,fill=region))+geom_col(position="dodge")+  
  theme(plot.title = element_text(color="black", size=16, face="bold.italic"),axis.title.x = element_text(color="black", size=16, face="bold"),
  axis.title.y = element_text(color="black", size=16, face="bold"))+theme(axis.text = element_text(color="black",size=12,face="bold"))+
  ggtitle("Percentage of foci >2500 sq.px") + theme(panel.background = element_blank())+labs(x="Antigen",y="Percentage")

ggplot(markers_small,aes(x=marker,y=percentage,fill=region))+geom_col(position="dodge")+  
  theme(plot.title = element_text(color="black", size=16, face="bold.italic"),axis.title.x = element_text(color="black", size=16, face="bold"),
        axis.title.y = element_text(color="black", size=16, face="bold"))+theme(axis.text = element_text(color="black",size=12,face="bold"))+
  ggtitle("Percentage of foci <2500 sq.px") + theme(panel.background = element_blank())+labs(x="Antigen",y="Percentage")
# change infil to margin in area 
area_cd31<-area_cd31 %>%
  mutate(across('region', str_replace, 'infil', 'margin'))
 
area_laminin<-area_laminin %>%
  mutate(across('region', str_replace, 'infil', 'margin'))
# CD31
wilcox.test(area ~ region,data= area_cd31)
t.test(area ~ region,data= area_cd31)

ggplot(data=area_cd31,aes(x=sqrt(area),fill=region)) + geom_density(alpha=0.4)+
  theme(plot.title = element_text(color="black", size=22, face="bold.italic"),
        axis.title.x = element_text(color="black", size=22, face="bold"),
        axis.title.y = element_text(color="black", size=22, face="bold"))+
  theme(axis.text = element_text(color="black",size=20,face="bold",angle=0))+
  ggtitle("distribution of area of CD31") + theme(panel.background = element_blank())+
  theme(legend.title = element_text(color = "black",size = 18),
        legend.text = element_text(color = "black",size=16))+xlim(-10,100)+
  # CD31
ggplot(data=area_cd31,aes(x=sqrt(area),fill=region)) + geom_density(alpha=0.4)+
  theme(plot.title = element_text(color="black", size=22, face="bold.italic"),
                      axis.title.x = element_text(color="black", size=22, face="bold"),
                      axis.title.y = element_text(color="black", size=22, face="bold"))+
  theme(axis.text = element_text(color="black",size=20,face="bold",angle=0))+
  ggtitle("distribution of area of CD31") + theme(panel.background = element_blank())+
  theme(legend.title = element_text(color = "black",size = 18),legend.text = element_text(color = "black",size=16))+
  xlim(-10,100)+
  annotate("text", x = 60, y = 0.08, label = "Wilcoxon rank sum test")+ 
  annotate("text", x = 60, y = 0.075, label = "                 median")+
  annotate("text", x = 60, y = 0.069, label = "           margin  19.9")+ 
  annotate("text", x = 60, y = 0.063, label = "           tumor   17.4")+
  annotate("text", x = 60, y = 0.058, label = "        (p-value=0.0001)")


area_cd31%>%group_by(region)%>%summarise(median=median(sqrt(area)))
# remove roi1 of margin before CD31 density plot calculation
area_cd31_edited<-area_cd31%>%filter(!(region=="margin" & ROI =="roi1"))
t.test(sqrt(area) ~ region,data= area_cd31_edited)
wilcox.test(sqrt(area) ~ region,data= area_cd31_edited)
area_cd31_edited%>%group_by(region)%>%summarise(median=median(sqrt(area)))
area_cd31_edited%>%group_by(region)%>%summarise(mean=mean(sqrt(area)))

area_cd31%>%group_by(region)%>%summarise(median=median(sqrt(area)))
area_cd31%>%group_by(region)%>%summarise(mean=mean(sqrt(area)))

area_cd31_edited%>%group_by(region)%>%summarise(median=median(area))
area_cd31_edited%>%group_by(region)%>%summarise(mean=mean(area))
area_cd31_edited%>%group_by(region)%>%summarise(mean=mean(area),sd=sd(area))
# change infil to margin in area 
area_markers<-area_markers %>%
  mutate(across('region', str_replace, 'infil', 'margin'))
area_markers_cd31<- area_markers %>% filter(marker=="CD31")

ggplot(area_markers_cd31,aes(x=category,y=fraction,color=region,fill=region))+
  geom_col(alpha=0.4)+  theme(plot.title = element_text(color="black", size=16, face="bold.italic"),
                              axis.title.x = element_text(color="black", size=16, face="bold"),
                              axis.title.y = element_text(color="black", size=16, face="bold"))+
  theme(axis.text = element_text(color="black",size=6,face="bold",angle=90))+labs(x="size stained area",y="fraction of total stained area")+facet_grid(region~marker)
# differential plot
diff_cd31<-area_markers_cd31%>%group_by(category)%>%mutate(col= fraction[region=="margin"]-fraction[region=="tumor"])
ggplot(diff_cd31[diff_cd31$region=="margin",],aes(x=category,y=col,fill = ifelse(col < 0, "blue", "yellow")))+ 
  geom_col()+coord_flip()+xlim(0,1000) +
  labs(x = "area", y = "change in proportion")+
  ggtitle("comparision of CD31 margin area over tumor area")+
  scale_fill_discrete(name = "region", labels = c("<tumor", ">tumor"))+
  theme(axis.text = element_text(color="black",size=14,face="bold"))+
  theme(plot.title = element_text(color="black", size=14, face="bold.italic"),
        axis.title.x = element_text(color="black", size=14, face="bold"),
        axis.title.y = element_text(color="black", size=14, face="bold"))
