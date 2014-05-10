# -*- coding: utf-8 -*-
"""
Created on Fri May  9 22:12:35 2014

@author: nairboon
"""

import sys
import numpy
import os


import pandas as pd
from ggplot import *

#runid = sys.argv[1]
#            print "analysis of ", runid

def average_run(acceptable, fn):
    c = pd.read_csv(fn)
    #c = pd.read_csv("mydata/"+runid+"/"+name)#,index_col=0)
    nc = c.columns.values
    nc[0] = "i"
    c.columns = nc
    
    jn = c.drop("i",1)
    avg = jn.mean(1)
    
    da = pd.DataFrame(avg)
    da["i"]=da.index
    
    
    da["Name"] = os.path.basename(fn)[:-4]
    print da

    n = len(jn.columns )
    ylabel = "Avg Reward over %d runs" % n
    avgr = ggplot(c,aes(x="i",y="0"))+geom_point()+geom_line()+ \
     xlab("Episodes") + ylab(ylabel) #stat_smooth(color='blue')
    #ggsave(avgr,fn+"_plot.png")
    #print avgr
    
    # converge indicator
    cv = []
    for run in jn.columns:
        found = False
        for i in range(0,len(jn[run])):
            if jn[run][i] >= acceptable:
                cv.append(i)
                found = True
                break
        if not found:
            cv.append(-9999)
    #print jn
    #print cv
    
    start = 2
    ns = []
    avgs = []
    for i in range(start-1,len(cv)):
        j = i+1
        avg  = numpy.mean(cv[:j])
        print avg,j, cv[:j]
        ns.append(j)
        avgs.append(avg)
        
    ap = pd.DataFrame({'n':ns,'avg':avgs})
    print ap
    ylabel = "Avg Reward over %d runs" % n
    avgdis = ggplot(ap,aes(x="n",y="avg"))+geom_point()+stat_smooth(color='blue')+ \
    labs(x="Numer of Experiment replicas",y="Avg Acceptable Solution",title="Required Replicas")    
    #print avgdis
     
    return (da,ap)
    

def main(parameters):
    label = sys.argv[-1]   # Sumatra appends the label to the command line
    subdir = os.path.join("mydata", label)
    #os.mkdir(subdir)

    res = {}
    an = []
    ax = []
    ay = []
    
    all_df = pd.DataFrame({"i":[],"Name":[]})
    
    final_df = pd.DataFrame({"Algorithm":[],"Task":[],"Steps":[]})

    for scenario in parameters["scenarios"]:
        res[scenario] = {}
        for algorithm in parameters["algorithms"]:
            name = scenario+"_"+algorithm
            fileid = "%s_%s.txt" % (scenario, algorithm)
            fn = os.path.join(subdir, fileid)
            da, ap = average_run(parameters["AcceptableScore"],fn)
            for i,r in ap.iterrows():
                an.append(name)
                ax.append(r["n"])
                ay.append(r["avg"])
                
            
            all_df = all_df.append(da)
            final_df = final_df.append(dict(Algorithm=algorithm,Task=scenario,Steps=ay[-1]),ignore_index=True)
                
    # showing that we have enough runs
    df = pd.DataFrame({"Name":an,"Runs":ax,"Avg":ay})
    print df
    p = ggplot(aes(x='Runs',y="Avg"), data=df) + geom_point() + geom_line()+ \
    facet_wrap("Name")
    ggsave(p,os.path.join(subdir, "avg_runs.png"))
    
    
    #ploting all runs
    all_df["y"] = all_df[0]
    #print all_df
    all_plot = ggplot(aes(x='i', y='y',colour="Name"), data=all_df) + geom_point() + geom_line()
    ggsave(all_plot,os.path.join(subdir, "all_runs.png"))

    #final comparison
    #do in R
    #print final_df
    final_df.to_csv(os.path.join(subdir, "final_comp.csv"),index=False)
    
    
    import subprocess
    proc = subprocess.Popen(['/usr/bin/Rscript','result.R',subdir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    print stdout,stderr
    proc.wait()
    print "done",subdir

    
    
            
            
    
import sumatra.parameters as p

parameter_file = sys.argv[1]
parameters = p.SimpleParameterSet(parameter_file)


main(parameters)

#average_run(-10,"mydata/yyy/d_bnrl.txt")
