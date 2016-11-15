PROC IMPORT 
        DATAFILE= "E:\UNCC\Fall 16\Machine Learning\project\Source_Files\Marcia\PD\Effectiveness_pd.xlsx"
        DBMS=xlsx 
     OUT=effect1 
      REPLACE;
    GETNAMES= YES;
    RUN;
	PROC IMPORT 
        DATAFILE= "E:\UNCC\Fall 16\Machine Learning\project\Source_Files\Marcia\PD\profile_metrics_pd.xlsx"
        DBMS=xlsx 
     OUT=promet1 
      REPLACE;
    GETNAMES= YES;
    RUN;
		PROC IMPORT 
        DATAFILE= "E:\UNCC\Fall 16\Machine Learning\project\Source_Files\Marcia\PD\Grades and Free lunch final.xlsx"
        DBMS=xlsx 
     OUT=grades 
      REPLACE;
    GETNAMES= YES;
    RUN;
		PROC IMPORT 
        DATAFILE= "E:\UNCC\Fall 16\Machine Learning\project\Source_Files\Marcia\PD\race_gender.xlsx"
        DBMS=xlsx 
     OUT=race_gen 
      REPLACE;
    GETNAMES= YES;
    RUN;
	PROC IMPORT 
        DATAFILE= "E:\UNCC\Fall 16\Machine Learning\project\Source_Files\Marcia\PD\personnel MP 201415.xlsx"
        DBMS=xlsx 
     OUT=personnel1 
      REPLACE;
    GETNAMES= YES;
    RUN;
PROC IMPORT 
        DATAFILE= "E:\UNCC\Fall 16\Machine Learning\project\Source_Files\Marcia\PD\environment MP 201415.xlsx"
        DBMS=xlsx 
     OUT=envi1 
      REPLACE;
    GETNAMES= YES;
    RUN;
	PROC IMPORT 
        DATAFILE= "E:\UNCC\Fall 16\Machine Learning\project\Source_Files\Marcia\PD\profile MP 201415.xlsx"
        DBMS=xlsx 
     OUT=profile1 
      REPLACE;
    GETNAMES= YES;
    RUN;
	data envi2; set envi1;
	media_age=input(avg_age_media_collection,4.);
	st_media_age=input(st_avg_age_media_collection,4.);
	books_per_student=input(books_per_student,4.);
	med_age_no=(2016-media_age);
	st_med_age_no=(2016-st_media_age);
	media_index_school=med_age_no/st_med_age_no;
	books_per_student=input(books_per_student,2.);
	st_books_per_student=input(st_books_per_student,2.);
	book_per_student_index=books_per_student/st_books_per_student;
	run;



	proc sort data=promet1 out=promet2 nodupkey;
by School_Name unit_code;
run;
proc sort data=personnel1 out=personnel2 nodupkey;
by School_Name unit_code;
run;
proc sort data=envi2 out=envi3 nodupkey;
by School_Name unit_code;
run;
proc sort data=profile1 out=profile2 nodupkey;
by School_Name unit_code;
run;

proc sort data=effect1 out=effect2 nodupkey;
by School_Name unit_code;
run;
proc sort data=race_gen out=race_gen1 nodupkey;
by School_Name unit_code;
run;
proc sort data=grades out=grades1 nodupkey;
by school_Name unit_code;
run;

proc sql;
   create table work.profilemet_effective as
   select a.School_Name,a.unit_code,a.Std1,a.Std2,a.Std3,a.Std4,a.Std5,a.Std,
b.State_size,b.size,b.Class_size_index,
   MalePer,PrSame,PrDiff,EDSPer,LEPPer,SWDPer,AIGPer,Grades,reduced_percent,Scores,
e.flicensed_teach_pct as NC_Licenced_teacher,e.tchyrs_0thru3_pct as t_03_exp,e.tchyrs_4thru10_pct as t_410_exp,
e.tchyrs_11plus_pct as t_11p_exp,e.advance_dgr_pct as adv_deg,e._1yr_tchr_trnovr_pct as one_yr_teacher_turnover,
e.lateral_teach_pct as lateral_entry_t,e.highqual_class_pct as highly_qualified_class_in_school,
f.avg_daily_attend_pct,f.crime_per_c_num as crime_per_100_stud,f.short_susp_per_c_num,
f.long_susp_per_c_num,f.expelled_per_c_num,f.stud_internet_comp_num,
f.book_per_student_index,f.media_index_school,g.type_cd,g.closed_ind,g.new_ind,
g.category_cd,g.school_type_txt,g.calendar_only_txt,g.title1_type_cd,g.clp_ind,
g.focus_clp_ind,g.summer_program_ind

from promet2 b,effect2 a,race_gen1 c,grades1 d,personnel1 e,envi3 f,profile2 g
where  d.unit_code=b.unit_code and d.unit_code=c.unit_code and d.unit_code=a.unit_code
and d.unit_code=e.unit_code and d.unit_code=f.unit_code and d.unit_code=g.unit_code;