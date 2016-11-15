PROC IMPORT 
        DATAFILE= "E:\UNCC\Fall 16\Machine Learning\project\Source_Files\Marcia\effectiveness MP 201415.xlsx"
        DBMS=xlsx 
     OUT=effect1 
      REPLACE;
    GETNAMES= YES;
    RUN;

proc sql;
   create table work.effect2 as
   select  unit_code,School_Name,level,year,STANDARD,input(percent,5.3) as percent
      from effect1 where percent is not missing and input(percent,5.3)<1;



proc sql;
   create table work.effect3 as
   select  unit_code,School_Name,level,year,STANDARD,sum(percent) as percent
      from effect2  
group by year,unit_code,School_Name,level,STANDARD   ;

data effect4; set effect3;
if(STANDARD ='Standard 1') then Standard1=percent; else Standard1=0 ;
if(STANDARD ='Standard 2') then Standard2=percent;else Standard2=0 ;
if(STANDARD ='Standard 3') then Standard3=percent;else Standard3=0 ;
if(STANDARD ='Standard 4') then Standard4=percent;else Standard4=0 ;
if(STANDARD ='Standard 5') then Standard5=percent;else Standard5=0 ;
Standard_avg=(Standard1+Standard2+Standard3+Standard4+Standard5)/5;
run;
 	

    proc sql;
   create table work.effect5 as
   select  School_Name,unit_code,sum(Standard1) as Std1,
sum(Standard2) as Std2,
sum(Standard3) as Std3,sum(Standard4) as Std4,sum(Standard5) as Std5,sum(Standard_avg) as Std
      from effect4 where level in ('Accomplished','Distinguished')
	  and unit_code not like ('%LEA')
group by School_Name,unit_code   ;