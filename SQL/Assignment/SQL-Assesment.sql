create database assessment;
use assessment;

create table city(
id int primary key auto_increment,
city_name varchar(50),
lat decimal(9,6),
long_ decimal(9,6),
country_id int 
);
create table customer(
id int primary key,
customer_name varchar(50),
 city_id INT,
customer_address varchar(100),
next_call_date date,
ts_inserted TIMESTAMP
);
CREATE TABLE country (
    id INT PRIMARY KEY,
    country_name VARCHAR(255),
    country_name_eng VARCHAR(255),
    country_code VARCHAR(3)
);

INSERT INTO city (id, city_name, lat, long_, country_id) VALUES
(1, 'Berlin', 52.520008, 13.404954, 1),
(2, 'Belgrade', 44.787197, 20.457273, 2),
(3, 'Zagreb', 45.815399, 15.966568, 3),
(4, 'New York', 40.730610, -73.935242, 4),
(5, 'Los Angeles', 34.052235, -118.243683, 4),
(6, 'Warsaw', 52.237049, 21.017532, 5);

INSERT INTO customer (id, customer_name, city_id, customer_address, next_call_date, ts_inserted) VALUES
(1, 'Jewelry Store', 4, 'Long Street 120', '2020-01-21', '2020-01-09 14:01:20.000'),
(2, 'Bakery', 1, 'Kufürstendamm 25', '2020-02-21', '2020-01-09 17:52:15.000'),
(3, 'Café', 1, 'Tauenrzenstraße 44', '2020-01-21', '2020-01-10 08:02:49.000'),
(4, 'Restaurant', 3, 'Ulica lipa 15', '2020-01-21', '2020-01-10 09:20:21.000');

INSERT INTO country (id, country_name, country_name_eng, country_code) VALUES
(1, 'Deutschland', 'Germany', 'DEU'),
(2, 'Srbija', 'Serbia', 'SRB'),
(3, 'Hrvatska', 'Croatia', 'HRV'),
(4, 'United States of America', 'United States of America', 'USA'),
(5, 'Polska', 'Poland', 'POL'),
(6, 'España', 'Spain', 'ESP'),
(7, 'Rossiya', 'Russia', 'RUS');


--  ---------- Task 1 --------------
SELECT c.id AS customer_id, c.customer_name, ci.city_name, co.country_name, c.customer_address,
c.next_call_date, c.ts_inserted FROM customer c 
JOIN city ci ON c.city_id = ci.id 
JOIN country co ON ci.country_id = co.id 
WHERE ci.city_name IN ('Berlin', 'Zagreb', 'New York');


SELECT co.country_name_eng AS country_name, ci.city_name,cu.customer_name 
FROM country co LEFT JOIN city ci ON co.id = ci.country_id 
LEFT JOIN customer cu ON ci.id = cu.city_id
ORDER BY co.country_name_eng, ci.city_name, cu.customer_name;


--  ---------- Task 2 --------------
SELECT co.country_name_eng AS country_name, ci.city_name,cu.customer_name 
FROM country co INNER JOIN city ci ON co.id = ci.country_id 
LEFT JOIN customer cu ON ci.id = cu.city_id 
ORDER BY co.country_name_eng, ci.city_name, cu.customer_name;