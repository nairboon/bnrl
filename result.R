library(ggplot2)

args <-commandArgs(trailingOnly = TRUE)

print(args)
path = args[1]
final_comp <- read.csv(paste(path,"/final_comp.csv", sep=""))
  #"~/dev/spyder/bnrl/mydata/yyy/final_comp.csv")

p <- ggplot(final_comp, aes(Algorithm, fill=Task,y=Steps)) + geom_bar(position="dodge",stat="identity")

#print(p)

ggsave("final_comparison.png",path=path)

print("Saved R plot")