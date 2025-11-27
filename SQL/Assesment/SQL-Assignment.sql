use hrdb;

-- 1) Display all information in the tables EMP and DEPT
select * from employees e
join departments d
on e.department_id = d.department_id;

-- 2) Display only the hire date and employee name for each employees.
select employee_id, concat(first_name, " ", last_name) as Name, hire_date
from employees;

-- 3) Display the ename concatenated with the job ID, separated by a comma
--  and space, and name the column Employee and Title
select concat(first_name,", ",job_id) as "Employee and Title"
from employees;

-- 4) Display the hire date, name and department number for all clerks.
select hire_date, concat(first_name, " ", last_name) as Name, department_id 
from employees; 

-- 5) Create a query to display all the data from the EMP table. Separate each column by a
-- comma. Name the column THE OUTPUT 
SELECT CONCAT(
    IFNULL(employee_id, 'NULL'), ', ',
    IFNULL(first_name, 'NULL'), ', ',
    IFNULL(last_name, 'NULL'), ', ',
    IFNULL(email, 'NULL'), ', ',
    IFNULL(phone_number, 'NULL'), ', ',
    IFNULL(hire_date, 'NULL'), ', ',
    IFNULL(job_id, 'NULL'), ', ',
    IFNULL(salary, 'NULL'), ', ',
    IFNULL(commission_pct, 'NULL'), ', ',
    IFNULL(manager_id, 'NULL'), ', ',
    IFNULL(department_id, 'NULL')
    
) AS "THE OUTPUT"
FROM employees;


-- 6) Display the names and salaries of all employees with a salary greater than 2000. 
select concat(first_name," ", last_name) as Name, salary 
from employees where salary > 2000;
 
-- 7) Display the names and dates of employees with the column headers "Name" and "Start Date"
select concat(first_name," ", last_name) as "Name", hire_date as "Start Date" from employees;

-- 8) Display the names and hire dates of all employees in the order they were hired
select concat(first_name," ", last_name) as "Name", hire_date from employees order by hire_date;

-- 9) Display the names and salaries of all employees in reverse salary order. 
select concat(first_name," ", last_name) as "Name", salary from employees order by salary desc;

-- 10) Display 'ename" and "deptno" who are all earned commission and display salary in reverse order. 
select  concat(first_name," ", last_name) as "Name", department_id, salary, commission_pct
from employees where commission_pct is not null 
order by salary desc; 

-- 11) Display the last name and job title of all employees who do not have a manager 
select e.last_name, j.job_title from employees e 
join jobs j on e.job_id = j.job_id 
where e.manager_id is null;

-- 12) Display the last name, job, and salary for all employees whose job is sales representative
-- or stock clerk and whose salary is not equal to $2,500, $3,500, or $5,000
select last_name, job_id, salary from employees
where job_id in ("ST_CLERK", "SA_REP") and salary not in (2500, 3500, 5000);


-- 13) Display the maximum, minimum and average salary and commission earned. 
select max(salary) as Max_Sal, min(salary) as Min_Sal, avg(salary) as Avg_Sal, commission_pct
from employees group by commission_pct;

-- 14) Display the department number, total salary payout and total commission payout for
-- each department.
select department_id, sum(salary) as Total_Sal, sum(commission_pct) as Total_Commission
from employees group by department_id;


-- 15) Display the department number and number of employees in each department. 
select department_id, count(employee_id) as "No of Emp"
from employees group by department_id;

-- 16) Display the department number and total salary of employees in each department
select department_id, sum(salary) as Total_Sal from employees group by department_id;

-- 17) Display the employee's name who doesn't earn a commission. Order the result set
-- without using the column name 
select concat(first_name, " " ,last_name) as Name
from employees where commission_pct is null order by 1;

-- 18) Display the employees name, department id and commission. If an Employee doesn't
-- earn the commission, then display as 'No commission'. Name the columns appropriately
select first_name, last_name, department_id,
case
when commission_pct is not null then commission_pct
when commission_pct is null then "No Commission"
end as Commission
from employees;

-- 19) Display the employee's name, salary and commission multiplied by 2. If an Employee
-- doesn't earn the commission, then display as 'No commission. Name the columns
-- appropriately
select first_name, last_name, salary, department_id,
case
when commission_pct is not null then commission_pct * 2
when commission_pct is null then "No Commission"
end as Commission
from employees;

-- 20) Display the employee's name, department id who have the first name same as another
-- employee in the same department
select e1.first_name, e1.department_id 
from employees e1 join employees e2
on e1.first_name = e2.first_name
and e1.department_id = e2.department_id 
and e1.employee_id <> e2.employee_id;

-- 21) Display the sum of salaries of the employees working under each Manager
select manager_id, sum(salary) as Salary from employees group by manager_id;

-- 22) Select the Managers name, the count of employees working under and the department
-- ID of the manager.
select m.manager_id, m.department_id, count(e.employee_id) as Count_emp 
from employees e join employees m 
on e.manager_id = m.employee_id
group by m.department_id, m.employee_id; 


-- 23) Select the employee name, department id, and the salary. Group the result with the
-- manager name and the employee last name should have second letter 'a! 
select e.first_name as emp_name,e.department_id, m.last_name as manager_name, e.salary
from employees e join employees m 
on e.manager_id = m.employee_id
where e.last_name like "_a%"
group by e.employee_id, e.department_id, e.salary, m.first_name;

-- 24) Display the average of sum of the salaries and group the result with the department id.
-- Order the result with the department id. 
with Total_Salary as (
select department_id, sum(salary) as Total_Sal
from employees 
group by department_id
)
select department_id, avg(Total_Sal) as Avg_Total_Sal from Total_Salary
group by department_id
order by department_id;


-- 25) Select the maximum salary of each department along with the department id 
select department_id, max(salary) as Max_Sal from employees
group by department_id;

-- 26) Display the commission, if not null display 10% of salary, if null display a default value 1
select commission_pct,
case
when commission_pct is not null then salary * 0.1
else 1
end as Commision_Ratio
from employees; 


-- 27) Write a query that displays the employee's last names only from the string's 2-5th
-- position with the first letter capitalized and all other letters lowercase, Give each column an
-- appropriate label. 

select concat(upper(substr(last_name, 2, 1)), lower(substr(last_name, 3,3))) as Formated_Lname from employees;


-- 28) Write a query that displays the employee's first name and last name along with a " in
-- between for e.g.: first name : Ram; last name : Kumar then Ram-Kumar. Also displays the
-- month on which the employee has joined.
select concat(first_name, '-', last_name) as Name, month(hire_date) as Joining_Month from employees
where first_name like '%a%' and last_name like '%a%';


-- 29) Write a query to display the employee's last name and if half of the salary is greater than
-- ten thousand then increase the salary by 10% else by 11.5% along with the bonus amount of
-- 1500 each. Provide each column an appropriate label. 
select last_name, 
case 
	when salary/2 > 10000 then (salary * 0.1)
	else (salary * 0.115)
end as Incresed_BY,
case 
	when salary/2 > 10000 then salary + (salary * 0.1) + 1500
	else salary + (salary * 0.115) + 1500
end as Total_Salary
from employees;


-- 30) Display the employee ID by Appending two zeros after 2nd digit and 'E' in the end,
-- department id, salary and the manager name all in Upper case, if the Manager name
-- consists of 'z' replace it with '$!
select concat(substr(e.employee_id, 1, 2), '00', substr(e.employee_id, -1), 'E') as Employee_ID,
e.department_id, e.salary,
upper(replace(m.last_name, 'z', '$')) as manager_name
from employees e
join employees m on e.manager_id = m.employee_id;


-- 31) Write a query that displays the employee's last names with the first letter capitalized and
-- all other letters lowercase, and the length of the names, for all employees whose name
-- starts with J, A, or M. Give each column an appropriate label. Sort the results by the
-- employees' last names 
select concat(upper(substr(last_name, 1, 1)), lower(substr(last_name, 2))) as Last_Name,
char_length(last_name) as Length_Lname 
from employees
WHERE last_name REGEXP '^[JAM]'
order by last_name;


-- 32) Create a query to display the last name and salary for all employees. Format the salary to
-- be 15 characters long, left-padded with $. Label the column SALARY
select last_name, lpad(salary, 15, '$') as SALARY from employees;


-- 33) Display the employee's name if it is a palindrome
select first_name, last_name from employees
where first_name = reverse(first_name) and
last_name = reverse(last_name);


-- 34) Display First names of all employees with initcaps.
select concat(upper(substr(first_name, 1, 1)), lower(substr(first_name,2))) as First_name from employees;

-- 35) From LOCATIONS table, extract the word between first and second space from the
-- STREET ADDRESS column.  
select street_address,
SUBSTRING_INDEX(SUBSTRING_INDEX(street_address, ' ', 2), ' ', -1)
from locations;


-- 36) Extract first letter from First Name column and append it with the Last Name. Also add
-- "@systechusa.com" at the end. Name the column as e-mail address. All characters should
-- be in lower case. Display this along with their First Name.
select first_name,
concat(lower(substr(first_name, 1, 1)), lower(substr(last_name,2)), '@systechusa.com') as Email 
from employees;


-- 37) Display the names and job titles of all employees with the same job as Trenna
select concat(e.first_name,' ', e.last_name) as Name, j.job_title
from employees e join jobs j 
on e.job_id = j.job_id
where j.job_id = (
select j.job_id from jobs j join employees e 
on e.job_id = j.job_id
where e.first_name = "Trenna");


-- 38) Display the names and department name of all employees working in the same city as Trenna. 
select concat(e.first_name,' ', e.last_name) as Name, d.department_name
from employees e join departments d
on e.department_id = d.department_id
where d.department_id = (
select d.department_id from departments d join employees e 
on e.department_id = d.department_id
where e.first_name = "Trenna");

-- 39) Display the name of the employee whose salary is the lowest. 
select concat(first_name, " ", last_name) as Name from employees
where salary = (select min(salary) from employees);

-- 40) Display the names of all employees except the lowest paid.
select concat(first_name, " ", last_name) as Name from employees
where salary != (select min(salary) from employees);


-- 41) Write a query to display the last name, department number, department name for all employees.
select e.last_name, d.department_id, d.department_name from employees e
join departments d on e.department_id = d.department_id;	

-- 42) Create a unique list of all jobs that are in department 40. Include the location of the
-- department in the output.
select j.job_id, j.job_title, d.location_id from employees e
join departments d on e.department_id = d.department_id
join jobs j on j.job_id = e.job_id
where d.department_id = 40;

-- 43) Write a query to display the employee last name,department name,location id and city of
-- all employees who earn commission.
select e.last_name, d.department_name, d.location_id, l.city from employees e
join departments d on e.department_id = d.department_id
join locations l on d.location_id = l.location_id 
where e.commission_pct is not null;

-- 44) Display the employee last name and department name of all employees who have an 'a'
-- in their last name 
select e.last_name, d.department_name from employees e
join departments d on e.department_id = d.department_id
where last_name like '%a%';

-- 45) Write a query to display the last name,job,department number and department name for
-- all employees who work in ATLANTA. 
select e.last_name, e.job_id, d.department_id, d.department_name from employees e
join departments d on e.department_id = d.department_id
join locations l on d.location_id = l.location_id
where l.city = 'ATLANTA'; 
 

-- 46) Display the employee last name and employee number along with their manager's last
-- name and manager number. 
select e.employee_id, e.last_name as Emp_Lname, m.manager_id, m.last_name as Manager_Lname from employees e
join employees m on e.manager_id = m.employee_id;


-- 47) Display the employee last name and employee number along with their manager's last
-- name and manager number (including the employees who have no manager).
select e.employee_id, e.last_name as Emp_Lname, m.manager_id, m.last_name as Manager_Lname from employees e
left join employees m on e.manager_id = m.employee_id;


-- 48) Create a query that displays employees last name,department number,and all the
-- employees who work in the same department as a given employee. 
select e1.last_name as Emp1_name, e1.department_id, e2.last_name as Emp2_name from employees e1
join employees e2 on e1.department_id = e2.department_id 
and e1.employee_id <> e2.employee_id
order by e1.employee_id, e1.department_id;

-- 49) Create a query that displays the name,job,department name,salary,grade for all
-- employees. Derive grade based on salary(>=50000=A, >=30000=B,<30000=C) 
select concat(e.first_name, ' ', e.last_name) as Name, e.job_id, d.department_name, e.salary,
case
when e.salary >= 50000 then 'A'
when e.salary >= 30000 then 'B'
when e.salary < 30000 then 'C'
end as grade from employees e 
join departments d on e.department_id = d.department_id;
 
 
-- 50) Display the names and hire date for all employees who were hired before their
-- managers along withe their manager names and hire date. Label the columns as Employee
-- name, emp_hire_date,manager name,man_hire_date
select e.last_name as Employee_name, e.hire_date as emp_hire_date,
m.last_name as Manager_name, m.hire_date as man_hire_date from employees e
join employees m on e.manager_id = m.employee_id
where e.hire_date < m.hire_date;


-- 51) Write a query to display the last name and hire date of any employee in the same
-- department as SALES.
select last_name, hire_date from employees
where department_id = (select department_id from departments where department_name = 'sales');


-- 52) Create a query to display the employee numbers and last names of all employees who
-- earn more than the average salary. Sort the results in ascending order of salary. 
select employee_id, last_name, salary from employees
where salary > (select avg(salary) from employees) order by salary;


-- 53) Write a query that displays the employee numbers and last names of all employees who
-- work in a department with any employee whose last name contains a' u 
select distinct e1.employee_id, e1.last_name from employees e1
join employees e2 on e1.department_id = e2.department_id
where e2.last_name like '%u%';


-- 54) Display the last name, department number, and job ID of all employees whose
-- department location is ATLANTA. 
select e.last_name, e.department_id, e.job_id from employees e
join departments d on e.department_id = d.department_id
join locations l on d.location_id = l.location_id
where l.city = 'ATLANTA';

-- 55) Display the last name and salary of every employee who reports to FILLMORE. 
select e.last_name, e.salary from employees e
join employees m on e.manager_id = m.employee_id
where m.last_name = 'FILLMORE';

-- 56) Display the department number, last name, and job ID for every employee in the
-- OPERATIONS department. 
select e.department_id, e.last_name, e.job_id from employees e
join departments d on e.department_id = d.department_id
where department_name = 'OPERATIONS';


-- 57) Modify the above query to display the employee numbers, last names, and salaries of all
-- employees who earn more than the average salary and who work in a department with any
-- employee with a 'u'in their name. 
select employee_id, last_name, salary from employees
where salary > (select avg(salary) from employees) 
and department_id in (select department_id from employees where first_name like '%u%');


-- 58) Display the names of all employees whose job title is the same as anyone in the sales dept.
select e.first_name, e.last_name from employees e 
where e.job_id in (select job_id from employees 
where department_id = (select department_id from departments where department_name = 'sales')); 


-- 59) Write a compound query to produce a list of employees showing raise percentages,
-- employee IDs, and salaries. Employees in department 1 and 3 are given a 5% raise,
-- employees in department 2 are given a 10% raise, employees in departments 4 and 5 are
-- given a 15% raise, and employees in department 6 are not given a raise.

SELECT employee_id, salary, '5%' AS raise_percentage, salary * 1.05 AS new_salary
FROM employees
WHERE department_id IN (10, 30)
UNION ALL
SELECT employee_id, salary, '10%' AS raise_percentage, salary * 1.10 AS new_salary
FROM employees
WHERE department_id = 20
UNION ALL
SELECT employee_id, salary, '15%' AS raise_percentage, salary * 1.15 AS new_salary
FROM employees
WHERE department_id IN (40, 50)
UNION ALL
SELECT employee_id, salary, '0%' AS raise_percentage, salary AS new_salary
FROM employees
WHERE department_id = 60;


-- 60) Write a query to display the top three earners in the EMPLOYEES table. Display their last
-- names and salaries. 
select last_name, salary from employees
order by salary desc limit 3;

-- 61) Display the names of all employees with their salary and commission earned. Employees
-- with a null commission should have O in the commission column 
select first_name, last_name, salary,
case 
when commission_pct is null then 0
else commission_pct * salary
end as commission_earned
from employees;


-- 62) Display the Managers (name) with top three salaries along with their salaries and
-- department information
select e.first_name, e.last_name, e.salary, d.department_name
from employees e
join departments d on e.department_id = d.department_id
where e.employee_id in (select manager_id from employees where manager_id is not null)
order by e.salary desc
limit 3;


-- --------------------------------  Date Questions ------------------------------------

create table if not exists date_function(
	Emp_ID int primary key not null auto_increment,
	hire_date date,
	Resignation_Date Date);
    
insert into date_function (hire_date, Resignation_Date) 
values
    ('2000-01-01', '2013-07-10'),
    ('2003-12-04', '2017-08-03'),
    ('2012-09-22', '2015-06-21'),
    ('2015-04-13', NULL),
    ('2016-06-03', NULL),
    ('2017-08-08', NULL),
    ('2016-11-13', NULL);
    
    
-- 1) Find the date difference between the hire date and resignation_date for all the
-- employees. Display in no. of days, months and year(1 year 3 months 5 days)
select hire_date,Resignation_Date, concat(
	timestampdiff(year,hire_date,Resignation_Date), ' Year ',
	timestampdiff(month,hire_date,Resignation_Date)%12, ' Month ',
	timestampdiff(day,hire_date,Resignation_Date), ' Days ') as Difference 
from date_function;


-- 2) Format the hire date as mm/dd/yyyy(09/22/2003) and resignation_date as mon dd,
-- yyyy(Aug 12th, 2004). Display the null as (DEC, 01th 1900) 
select date_format(hire_date,'%m/%d/%y') as hire_date,
ifNULL(date_format(Resignation_Date,'%b %e, %y') ,
'Dec 01st, 1900') as Regination_date from date_function;


-- 3) Calcuate experience of the employee till date in Years and months(example 1 year and 3 months)
SELECT 
    CONCAT(
        IFNULL(TIMESTAMPDIFF(YEAR, hire_date, Resignation_Date), TIMESTAMPDIFF(YEAR, hire_date, CURDATE())), " year ",
        IFNULL(TIMESTAMPDIFF(MONTH, hire_date, Resignation_Date) % 12, TIMESTAMPDIFF(MONTH, hire_date, CURDATE()) % 12), " month"
    ) AS experience
FROM date_function;
 
 
-- 4) Display the count of days in the previous quarter\
SELECT 
    TIMESTAMPDIFF(DAY, 
        DATE_SUB(DATE_FORMAT(CURDATE(), '%Y-%m-01'), INTERVAL 3 MONTH), 
        DATE_SUB(DATE_FORMAT(CURDATE(), '%Y-%m-01'), INTERVAL 0 MONTH)
    ) AS days_in_previous_quarter;


-- 5) Fetch the previous Quarter's second week's first day's date
SELECT 
DATE_ADD(
	DATE_SUB(DATE_FORMAT(CURDATE(), '%Y-%m-01'), INTERVAL 3 MONTH), 
	INTERVAL 7 DAY) AS second_week_first_day_previous_quarter;


-- 6) Fetch the financial year's 15th week's dates (Format: Mon DD YYYY)
SELECT 
DATE_FORMAT(DATE_ADD(
	STR_TO_DATE(CONCAT(YEAR(CURDATE()), '-04-01'), '%Y-%m-%d'), 
	INTERVAL (15 - 1) WEEK),'%a %d %Y') 
	AS first_day_15th_week,
DATE_FORMAT(DATE_ADD(DATE_ADD(
	STR_TO_DATE(CONCAT(YEAR(CURDATE()), '-04-01'), '%Y-%m-%d'), 
	INTERVAL (15 - 1) WEEK),INTERVAL 6 DAY), '%a %d %Y') 
	AS last_day_15th_week;
    
    
-- 7) Find out the date that corresponds to the last Saturday of January, 2015 using with clause
WITH last_day_january_2015 AS (
SELECT '2015-01-31' AS last_day)
SELECT DATE_SUB(last_day, INTERVAL (DAYOFWEEK(last_day) + 1) % 7 DAY) AS last_saturday_january_2015
FROM last_day_january_2015;

