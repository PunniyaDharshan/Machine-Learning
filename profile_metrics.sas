PROC IMPORT 
        DATAFILE= "E:\UNCC\Fall 16\Machine Learning\project\Source_Files\Marcia\profile-metrics MP 201415.xlsx"
        DBMS=xlsx 
     OUT=promet1 
      REPLACE;
    GETNAMES= YES;
    RUN;

data promet2; set promet1;
size1=input(size,8.);
drop size;

run;

data promet3;set promet2;
if size1=. then size1=0;else size1=size1;
run;

proc sql;
   create table work.promet4 as
   select  round(avg(st_size),.01) as State_size,round(avg(size1),.01) as size,School_Name, unit_code
 	
      from promet3 group by School_Name,unit_code;

	  proc sql;
   create table work.promet5 as
   select  round(avg(State_size),.01) as State_size,round(avg(size),.01) as size,
round(avg(size),.01)/round(avg(State_size),.01) as Class_size_index,
School_Name, unit_code
 	
      from promet4 group by School_Name,unit_code;





