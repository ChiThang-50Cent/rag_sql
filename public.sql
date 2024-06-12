/*
 Navicat Premium Data Transfer

 Source Server         : pagila
 Source Server Type    : PostgreSQL
 Source Server Version : 160003 (160003)
 Source Host           : ep-jolly-fog-a73ggts7.ap-southeast-2.aws.neon.tech:5432
 Source Catalog        : pagila
 Source Schema         : public

 Target Server Type    : PostgreSQL
 Target Server Version : 160003 (160003)
 File Encoding         : 65001

 Date: 12/06/2024 10:17:48
*/


-- ----------------------------
-- Type structure for mpaa_rating
-- ----------------------------
DROP TYPE IF EXISTS "public"."mpaa_rating";
CREATE TYPE "public"."mpaa_rating" AS ENUM (
  'G',
  'PG',
  'PG-13',
  'R',
  'NC-17'
);
ALTER TYPE "public"."mpaa_rating" OWNER TO "paliga_owner";

-- ----------------------------
-- Sequence structure for actor_actor_id_seq
-- ----------------------------
DROP SEQUENCE IF EXISTS "public"."actor_actor_id_seq";
CREATE SEQUENCE "public"."actor_actor_id_seq" 
INCREMENT 1
MINVALUE  1
MAXVALUE 9223372036854775807
START 1
CACHE 1;

-- ----------------------------
-- Sequence structure for address_address_id_seq
-- ----------------------------
DROP SEQUENCE IF EXISTS "public"."address_address_id_seq";
CREATE SEQUENCE "public"."address_address_id_seq" 
INCREMENT 1
MINVALUE  1
MAXVALUE 9223372036854775807
START 1
CACHE 1;

-- ----------------------------
-- Sequence structure for category_category_id_seq
-- ----------------------------
DROP SEQUENCE IF EXISTS "public"."category_category_id_seq";
CREATE SEQUENCE "public"."category_category_id_seq" 
INCREMENT 1
MINVALUE  1
MAXVALUE 9223372036854775807
START 1
CACHE 1;

-- ----------------------------
-- Sequence structure for city_city_id_seq
-- ----------------------------
DROP SEQUENCE IF EXISTS "public"."city_city_id_seq";
CREATE SEQUENCE "public"."city_city_id_seq" 
INCREMENT 1
MINVALUE  1
MAXVALUE 9223372036854775807
START 1
CACHE 1;

-- ----------------------------
-- Sequence structure for country_country_id_seq
-- ----------------------------
DROP SEQUENCE IF EXISTS "public"."country_country_id_seq";
CREATE SEQUENCE "public"."country_country_id_seq" 
INCREMENT 1
MINVALUE  1
MAXVALUE 9223372036854775807
START 1
CACHE 1;

-- ----------------------------
-- Sequence structure for customer_customer_id_seq
-- ----------------------------
DROP SEQUENCE IF EXISTS "public"."customer_customer_id_seq";
CREATE SEQUENCE "public"."customer_customer_id_seq" 
INCREMENT 1
MINVALUE  1
MAXVALUE 9223372036854775807
START 1
CACHE 1;

-- ----------------------------
-- Sequence structure for film_film_id_seq
-- ----------------------------
DROP SEQUENCE IF EXISTS "public"."film_film_id_seq";
CREATE SEQUENCE "public"."film_film_id_seq" 
INCREMENT 1
MINVALUE  1
MAXVALUE 9223372036854775807
START 1
CACHE 1;

-- ----------------------------
-- Sequence structure for inventory_inventory_id_seq
-- ----------------------------
DROP SEQUENCE IF EXISTS "public"."inventory_inventory_id_seq";
CREATE SEQUENCE "public"."inventory_inventory_id_seq" 
INCREMENT 1
MINVALUE  1
MAXVALUE 9223372036854775807
START 1
CACHE 1;

-- ----------------------------
-- Sequence structure for language_language_id_seq
-- ----------------------------
DROP SEQUENCE IF EXISTS "public"."language_language_id_seq";
CREATE SEQUENCE "public"."language_language_id_seq" 
INCREMENT 1
MINVALUE  1
MAXVALUE 9223372036854775807
START 1
CACHE 1;

-- ----------------------------
-- Sequence structure for payment_payment_id_seq
-- ----------------------------
DROP SEQUENCE IF EXISTS "public"."payment_payment_id_seq";
CREATE SEQUENCE "public"."payment_payment_id_seq" 
INCREMENT 1
MINVALUE  1
MAXVALUE 9223372036854775807
START 1
CACHE 1;

-- ----------------------------
-- Sequence structure for rental_rental_id_seq
-- ----------------------------
DROP SEQUENCE IF EXISTS "public"."rental_rental_id_seq";
CREATE SEQUENCE "public"."rental_rental_id_seq" 
INCREMENT 1
MINVALUE  1
MAXVALUE 9223372036854775807
START 1
CACHE 1;

-- ----------------------------
-- Sequence structure for staff_staff_id_seq
-- ----------------------------
DROP SEQUENCE IF EXISTS "public"."staff_staff_id_seq";
CREATE SEQUENCE "public"."staff_staff_id_seq" 
INCREMENT 1
MINVALUE  1
MAXVALUE 9223372036854775807
START 1
CACHE 1;

-- ----------------------------
-- Sequence structure for store_store_id_seq
-- ----------------------------
DROP SEQUENCE IF EXISTS "public"."store_store_id_seq";
CREATE SEQUENCE "public"."store_store_id_seq" 
INCREMENT 1
MINVALUE  1
MAXVALUE 9223372036854775807
START 1
CACHE 1;

-- ----------------------------
-- Table structure for actor
-- ----------------------------
DROP TABLE IF EXISTS "public"."actor";
CREATE TABLE "public"."actor" (
  "actor_id" int4 NOT NULL DEFAULT nextval('actor_actor_id_seq'::regclass),
  "first_name" text COLLATE "pg_catalog"."default" NOT NULL,
  "last_name" text COLLATE "pg_catalog"."default" NOT NULL,
  "last_update" timestamptz(6) NOT NULL DEFAULT now()
)
;

-- ----------------------------
-- Table structure for address
-- ----------------------------
DROP TABLE IF EXISTS "public"."address";
CREATE TABLE "public"."address" (
  "address_id" int4 NOT NULL DEFAULT nextval('address_address_id_seq'::regclass),
  "address" text COLLATE "pg_catalog"."default" NOT NULL,
  "address2" text COLLATE "pg_catalog"."default",
  "district" text COLLATE "pg_catalog"."default" NOT NULL,
  "city_id" int4 NOT NULL,
  "postal_code" text COLLATE "pg_catalog"."default",
  "phone" text COLLATE "pg_catalog"."default" NOT NULL,
  "last_update" timestamptz(6) NOT NULL DEFAULT now()
)
;

-- ----------------------------
-- Table structure for category
-- ----------------------------
DROP TABLE IF EXISTS "public"."category";
CREATE TABLE "public"."category" (
  "category_id" int4 NOT NULL DEFAULT nextval('category_category_id_seq'::regclass),
  "name" text COLLATE "pg_catalog"."default" NOT NULL,
  "last_update" timestamptz(6) NOT NULL DEFAULT now()
)
;

-- ----------------------------
-- Table structure for city
-- ----------------------------
DROP TABLE IF EXISTS "public"."city";
CREATE TABLE "public"."city" (
  "city_id" int4 NOT NULL DEFAULT nextval('city_city_id_seq'::regclass),
  "city" text COLLATE "pg_catalog"."default" NOT NULL,
  "country_id" int4 NOT NULL,
  "last_update" timestamptz(6) NOT NULL DEFAULT now()
)
;

-- ----------------------------
-- Table structure for country
-- ----------------------------
DROP TABLE IF EXISTS "public"."country";
CREATE TABLE "public"."country" (
  "country_id" int4 NOT NULL DEFAULT nextval('country_country_id_seq'::regclass),
  "country" text COLLATE "pg_catalog"."default" NOT NULL,
  "last_update" timestamptz(6) NOT NULL DEFAULT now()
)
;

-- ----------------------------
-- Table structure for customer
-- ----------------------------
DROP TABLE IF EXISTS "public"."customer";
CREATE TABLE "public"."customer" (
  "customer_id" int4 NOT NULL DEFAULT nextval('customer_customer_id_seq'::regclass),
  "store_id" int4 NOT NULL,
  "first_name" text COLLATE "pg_catalog"."default" NOT NULL,
  "last_name" text COLLATE "pg_catalog"."default" NOT NULL,
  "email" text COLLATE "pg_catalog"."default",
  "address_id" int4 NOT NULL,
  "activebool" bool NOT NULL DEFAULT true,
  "create_date" date NOT NULL DEFAULT CURRENT_DATE,
  "last_update" timestamptz(6) DEFAULT now(),
  "active" int4
)
;

-- ----------------------------
-- Table structure for film
-- ----------------------------
DROP TABLE IF EXISTS "public"."film";
CREATE TABLE "public"."film" (
  "film_id" int4 NOT NULL DEFAULT nextval('film_film_id_seq'::regclass),
  "title" text COLLATE "pg_catalog"."default" NOT NULL,
  "description" text COLLATE "pg_catalog"."default",
  "release_year" "public"."year",
  "language_id" int4 NOT NULL,
  "original_language_id" int4,
  "rental_duration" int2 NOT NULL DEFAULT 3,
  "rental_rate" numeric(4,2) NOT NULL DEFAULT 4.99,
  "length" int2,
  "replacement_cost" numeric(5,2) NOT NULL DEFAULT 19.99,
  "rating" "public"."mpaa_rating" DEFAULT 'G'::mpaa_rating,
  "last_update" timestamptz(6) NOT NULL DEFAULT now(),
  "special_features" text[] COLLATE "pg_catalog"."default",
  "fulltext" tsvector NOT NULL
)
;

-- ----------------------------
-- Table structure for film_actor
-- ----------------------------
DROP TABLE IF EXISTS "public"."film_actor";
CREATE TABLE "public"."film_actor" (
  "actor_id" int4 NOT NULL,
  "film_id" int4 NOT NULL,
  "last_update" timestamptz(6) NOT NULL DEFAULT now()
)
;

-- ----------------------------
-- Table structure for film_category
-- ----------------------------
DROP TABLE IF EXISTS "public"."film_category";
CREATE TABLE "public"."film_category" (
  "film_id" int4 NOT NULL,
  "category_id" int4 NOT NULL,
  "last_update" timestamptz(6) NOT NULL DEFAULT now()
)
;

-- ----------------------------
-- Table structure for inventory
-- ----------------------------
DROP TABLE IF EXISTS "public"."inventory";
CREATE TABLE "public"."inventory" (
  "inventory_id" int4 NOT NULL DEFAULT nextval('inventory_inventory_id_seq'::regclass),
  "film_id" int4 NOT NULL,
  "store_id" int4 NOT NULL,
  "last_update" timestamptz(6) NOT NULL DEFAULT now()
)
;

-- ----------------------------
-- Table structure for language
-- ----------------------------
DROP TABLE IF EXISTS "public"."language";
CREATE TABLE "public"."language" (
  "language_id" int4 NOT NULL DEFAULT nextval('language_language_id_seq'::regclass),
  "name" char(20) COLLATE "pg_catalog"."default" NOT NULL,
  "last_update" timestamptz(6) NOT NULL DEFAULT now()
)
;

-- ----------------------------
-- Table structure for payment_p2022_01
-- ----------------------------
DROP TABLE IF EXISTS "public"."payment_p2022_01";
CREATE TABLE "public"."payment_p2022_01" (
  "payment_id" int4 NOT NULL DEFAULT nextval('payment_payment_id_seq'::regclass),
  "customer_id" int4 NOT NULL,
  "staff_id" int4 NOT NULL,
  "rental_id" int4 NOT NULL,
  "amount" numeric(5,2) NOT NULL,
  "payment_date" timestamptz(6) NOT NULL
)
;

-- ----------------------------
-- Table structure for payment_p2022_02
-- ----------------------------
DROP TABLE IF EXISTS "public"."payment_p2022_02";
CREATE TABLE "public"."payment_p2022_02" (
  "payment_id" int4 NOT NULL DEFAULT nextval('payment_payment_id_seq'::regclass),
  "customer_id" int4 NOT NULL,
  "staff_id" int4 NOT NULL,
  "rental_id" int4 NOT NULL,
  "amount" numeric(5,2) NOT NULL,
  "payment_date" timestamptz(6) NOT NULL
)
;

-- ----------------------------
-- Table structure for payment_p2022_03
-- ----------------------------
DROP TABLE IF EXISTS "public"."payment_p2022_03";
CREATE TABLE "public"."payment_p2022_03" (
  "payment_id" int4 NOT NULL DEFAULT nextval('payment_payment_id_seq'::regclass),
  "customer_id" int4 NOT NULL,
  "staff_id" int4 NOT NULL,
  "rental_id" int4 NOT NULL,
  "amount" numeric(5,2) NOT NULL,
  "payment_date" timestamptz(6) NOT NULL
)
;

-- ----------------------------
-- Table structure for payment_p2022_04
-- ----------------------------
DROP TABLE IF EXISTS "public"."payment_p2022_04";
CREATE TABLE "public"."payment_p2022_04" (
  "payment_id" int4 NOT NULL DEFAULT nextval('payment_payment_id_seq'::regclass),
  "customer_id" int4 NOT NULL,
  "staff_id" int4 NOT NULL,
  "rental_id" int4 NOT NULL,
  "amount" numeric(5,2) NOT NULL,
  "payment_date" timestamptz(6) NOT NULL
)
;

-- ----------------------------
-- Table structure for payment_p2022_05
-- ----------------------------
DROP TABLE IF EXISTS "public"."payment_p2022_05";
CREATE TABLE "public"."payment_p2022_05" (
  "payment_id" int4 NOT NULL DEFAULT nextval('payment_payment_id_seq'::regclass),
  "customer_id" int4 NOT NULL,
  "staff_id" int4 NOT NULL,
  "rental_id" int4 NOT NULL,
  "amount" numeric(5,2) NOT NULL,
  "payment_date" timestamptz(6) NOT NULL
)
;

-- ----------------------------
-- Table structure for payment_p2022_06
-- ----------------------------
DROP TABLE IF EXISTS "public"."payment_p2022_06";
CREATE TABLE "public"."payment_p2022_06" (
  "payment_id" int4 NOT NULL DEFAULT nextval('payment_payment_id_seq'::regclass),
  "customer_id" int4 NOT NULL,
  "staff_id" int4 NOT NULL,
  "rental_id" int4 NOT NULL,
  "amount" numeric(5,2) NOT NULL,
  "payment_date" timestamptz(6) NOT NULL
)
;

-- ----------------------------
-- Table structure for payment_p2022_07
-- ----------------------------
DROP TABLE IF EXISTS "public"."payment_p2022_07";
CREATE TABLE "public"."payment_p2022_07" (
  "payment_id" int4 NOT NULL DEFAULT nextval('payment_payment_id_seq'::regclass),
  "customer_id" int4 NOT NULL,
  "staff_id" int4 NOT NULL,
  "rental_id" int4 NOT NULL,
  "amount" numeric(5,2) NOT NULL,
  "payment_date" timestamptz(6) NOT NULL
)
;

-- ----------------------------
-- Table structure for rental
-- ----------------------------
DROP TABLE IF EXISTS "public"."rental";
CREATE TABLE "public"."rental" (
  "rental_id" int4 NOT NULL DEFAULT nextval('rental_rental_id_seq'::regclass),
  "rental_date" timestamptz(6) NOT NULL,
  "inventory_id" int4 NOT NULL,
  "customer_id" int4 NOT NULL,
  "return_date" timestamptz(6),
  "staff_id" int4 NOT NULL,
  "last_update" timestamptz(6) NOT NULL DEFAULT now()
)
;

-- ----------------------------
-- Table structure for staff
-- ----------------------------
DROP TABLE IF EXISTS "public"."staff";
CREATE TABLE "public"."staff" (
  "staff_id" int4 NOT NULL DEFAULT nextval('staff_staff_id_seq'::regclass),
  "first_name" text COLLATE "pg_catalog"."default" NOT NULL,
  "last_name" text COLLATE "pg_catalog"."default" NOT NULL,
  "address_id" int4 NOT NULL,
  "email" text COLLATE "pg_catalog"."default",
  "store_id" int4 NOT NULL,
  "active" bool NOT NULL DEFAULT true,
  "username" text COLLATE "pg_catalog"."default" NOT NULL,
  "password" text COLLATE "pg_catalog"."default",
  "last_update" timestamptz(6) NOT NULL DEFAULT now(),
  "picture" bytea
)
;

-- ----------------------------
-- Table structure for store
-- ----------------------------
DROP TABLE IF EXISTS "public"."store";
CREATE TABLE "public"."store" (
  "store_id" int4 NOT NULL DEFAULT nextval('store_store_id_seq'::regclass),
  "manager_staff_id" int4 NOT NULL,
  "address_id" int4 NOT NULL,
  "last_update" timestamptz(6) NOT NULL DEFAULT now()
)
;

-- ----------------------------
-- Table structure for payment
-- ----------------------------
DROP TABLE IF EXISTS "public"."payment";
CREATE TABLE "public"."payment" (
  "payment_id" int4 NOT NULL DEFAULT nextval('payment_payment_id_seq'::regclass),
  "customer_id" int4 NOT NULL,
  "staff_id" int4 NOT NULL,
  "rental_id" int4 NOT NULL,
  "amount" numeric(5,2) NOT NULL,
  "payment_date" timestamptz(6) NOT NULL
)
PARTITION BY RANGE (
  "payment_date" "pg_catalog"."timestamptz_ops"
)
;
ALTER TABLE "public"."payment" ATTACH PARTITION "public"."payment_p2022_01" FOR VALUES FROM (
'2022-01-01 00:00:00+00'
) TO (
'2022-02-01 00:00:00+00'
)
;
ALTER TABLE "public"."payment" ATTACH PARTITION "public"."payment_p2022_02" FOR VALUES FROM (
'2022-02-01 00:00:00+00'
) TO (
'2022-03-01 00:00:00+00'
)
;
ALTER TABLE "public"."payment" ATTACH PARTITION "public"."payment_p2022_03" FOR VALUES FROM (
'2022-03-01 00:00:00+00'
) TO (
'2022-04-01 00:00:00+00'
)
;
ALTER TABLE "public"."payment" ATTACH PARTITION "public"."payment_p2022_04" FOR VALUES FROM (
'2022-04-01 00:00:00+00'
) TO (
'2022-05-01 00:00:00+00'
)
;
ALTER TABLE "public"."payment" ATTACH PARTITION "public"."payment_p2022_05" FOR VALUES FROM (
'2022-05-01 00:00:00+00'
) TO (
'2022-06-01 00:00:00+00'
)
;
ALTER TABLE "public"."payment" ATTACH PARTITION "public"."payment_p2022_06" FOR VALUES FROM (
'2022-06-01 00:00:00+00'
) TO (
'2022-07-01 00:00:00+00'
)
;
ALTER TABLE "public"."payment" ATTACH PARTITION "public"."payment_p2022_07" FOR VALUES FROM (
'2022-07-01 00:00:00+00'
) TO (
'2022-08-01 00:00:00+00'
)
;

-- ----------------------------
-- Function structure for _group_concat
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_group_concat"(text, text);
CREATE OR REPLACE FUNCTION "public"."_group_concat"(text, text)
  RETURNS "pg_catalog"."text" AS $BODY$
SELECT CASE
  WHEN $2 IS NULL THEN $1
  WHEN $1 IS NULL THEN $2
  ELSE $1 || ', ' || $2
END
$BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;

-- ----------------------------
-- Function structure for film_in_stock
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."film_in_stock"("p_film_id" int4, "p_store_id" int4, OUT "p_film_count" int4);
CREATE OR REPLACE FUNCTION "public"."film_in_stock"(IN "p_film_id" int4, IN "p_store_id" int4, OUT "p_film_count" int4)
  RETURNS SETOF "pg_catalog"."int4" AS $BODY$
     SELECT inventory_id
     FROM inventory
     WHERE film_id = $1
     AND store_id = $2
     AND inventory_in_stock(inventory_id);
$BODY$
  LANGUAGE sql VOLATILE
  COST 100
  ROWS 1000;

-- ----------------------------
-- Function structure for film_not_in_stock
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."film_not_in_stock"("p_film_id" int4, "p_store_id" int4, OUT "p_film_count" int4);
CREATE OR REPLACE FUNCTION "public"."film_not_in_stock"(IN "p_film_id" int4, IN "p_store_id" int4, OUT "p_film_count" int4)
  RETURNS SETOF "pg_catalog"."int4" AS $BODY$
    SELECT inventory_id
    FROM inventory
    WHERE film_id = $1
    AND store_id = $2
    AND NOT inventory_in_stock(inventory_id);
$BODY$
  LANGUAGE sql VOLATILE
  COST 100
  ROWS 1000;

-- ----------------------------
-- Function structure for get_customer_balance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."get_customer_balance"("p_customer_id" int4, "p_effective_date" timestamptz);
CREATE OR REPLACE FUNCTION "public"."get_customer_balance"("p_customer_id" int4, "p_effective_date" timestamptz)
  RETURNS "pg_catalog"."numeric" AS $BODY$
       --#OK, WE NEED TO CALCULATE THE CURRENT BALANCE GIVEN A CUSTOMER_ID AND A DATE
       --#THAT WE WANT THE BALANCE TO BE EFFECTIVE FOR. THE BALANCE IS:
       --#   1) RENTAL FEES FOR ALL PREVIOUS RENTALS
       --#   2) ONE DOLLAR FOR EVERY DAY THE PREVIOUS RENTALS ARE OVERDUE
       --#   3) IF A FILM IS MORE THAN RENTAL_DURATION * 2 OVERDUE, CHARGE THE REPLACEMENT_COST
       --#   4) SUBTRACT ALL PAYMENTS MADE BEFORE THE DATE SPECIFIED
DECLARE
    v_rentfees DECIMAL(5,2); --#FEES PAID TO RENT THE VIDEOS INITIALLY
    v_overfees INTEGER;      --#LATE FEES FOR PRIOR RENTALS
    v_payments DECIMAL(5,2); --#SUM OF PAYMENTS MADE PREVIOUSLY
BEGIN
    SELECT COALESCE(SUM(film.rental_rate),0) INTO v_rentfees
    FROM film, inventory, rental
    WHERE film.film_id = inventory.film_id
      AND inventory.inventory_id = rental.inventory_id
      AND rental.rental_date <= p_effective_date
      AND rental.customer_id = p_customer_id;

    SELECT COALESCE(SUM(IF((rental.return_date - rental.rental_date) > (film.rental_duration * '1 day'::interval),
        ((rental.return_date - rental.rental_date) - (film.rental_duration * '1 day'::interval)),0)),0) INTO v_overfees
    FROM rental, inventory, film
    WHERE film.film_id = inventory.film_id
      AND inventory.inventory_id = rental.inventory_id
      AND rental.rental_date <= p_effective_date
      AND rental.customer_id = p_customer_id;

    SELECT COALESCE(SUM(payment.amount),0) INTO v_payments
    FROM payment
    WHERE payment.payment_date <= p_effective_date
    AND payment.customer_id = p_customer_id;

    RETURN v_rentfees + v_overfees - v_payments;
END
$BODY$
  LANGUAGE plpgsql VOLATILE
  COST 100;

-- ----------------------------
-- Function structure for inventory_held_by_customer
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."inventory_held_by_customer"("p_inventory_id" int4);
CREATE OR REPLACE FUNCTION "public"."inventory_held_by_customer"("p_inventory_id" int4)
  RETURNS "pg_catalog"."int4" AS $BODY$
DECLARE
    v_customer_id INTEGER;
BEGIN

  SELECT customer_id INTO v_customer_id
  FROM rental
  WHERE return_date IS NULL
  AND inventory_id = p_inventory_id;

  RETURN v_customer_id;
END $BODY$
  LANGUAGE plpgsql VOLATILE
  COST 100;

-- ----------------------------
-- Function structure for inventory_in_stock
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."inventory_in_stock"("p_inventory_id" int4);
CREATE OR REPLACE FUNCTION "public"."inventory_in_stock"("p_inventory_id" int4)
  RETURNS "pg_catalog"."bool" AS $BODY$
DECLARE
    v_rentals INTEGER;
    v_out     INTEGER;
BEGIN
    -- AN ITEM IS IN-STOCK IF THERE ARE EITHER NO ROWS IN THE rental TABLE
    -- FOR THE ITEM OR ALL ROWS HAVE return_date POPULATED

    SELECT count(*) INTO v_rentals
    FROM rental
    WHERE inventory_id = p_inventory_id;

    IF v_rentals = 0 THEN
      RETURN TRUE;
    END IF;

    SELECT COUNT(rental_id) INTO v_out
    FROM inventory LEFT JOIN rental USING(inventory_id)
    WHERE inventory.inventory_id = p_inventory_id
    AND rental.return_date IS NULL;

    IF v_out > 0 THEN
      RETURN FALSE;
    ELSE
      RETURN TRUE;
    END IF;
END $BODY$
  LANGUAGE plpgsql VOLATILE
  COST 100;

-- ----------------------------
-- Function structure for last_day
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."last_day"(timestamptz);
CREATE OR REPLACE FUNCTION "public"."last_day"(timestamptz)
  RETURNS "pg_catalog"."date" AS $BODY$
  SELECT CASE
    WHEN EXTRACT(MONTH FROM $1) = 12 THEN
      (((EXTRACT(YEAR FROM $1) + 1) operator(pg_catalog.||) '-01-01')::date - INTERVAL '1 day')::date
    ELSE
      ((EXTRACT(YEAR FROM $1) operator(pg_catalog.||) '-' operator(pg_catalog.||) (EXTRACT(MONTH FROM $1) + 1) operator(pg_catalog.||) '-01')::date - INTERVAL '1 day')::date
    END
$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;

-- ----------------------------
-- Function structure for last_updated
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."last_updated"();
CREATE OR REPLACE FUNCTION "public"."last_updated"()
  RETURNS "pg_catalog"."trigger" AS $BODY$
BEGIN
    NEW.last_update = CURRENT_TIMESTAMP;
    RETURN NEW;
END $BODY$
  LANGUAGE plpgsql VOLATILE
  COST 100;

-- ----------------------------
-- Function structure for rewards_report
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."rewards_report"("min_monthly_purchases" int4, "min_dollar_amount_purchased" numeric);
CREATE OR REPLACE FUNCTION "public"."rewards_report"("min_monthly_purchases" int4, "min_dollar_amount_purchased" numeric)
  RETURNS SETOF "public"."customer" AS $BODY$
DECLARE
    last_month_start DATE;
    last_month_end DATE;
rr RECORD;
tmpSQL TEXT;
BEGIN

    /* Some sanity checks... */
    IF min_monthly_purchases = 0 THEN
        RAISE EXCEPTION 'Minimum monthly purchases parameter must be > 0';
    END IF;
    IF min_dollar_amount_purchased = 0.00 THEN
        RAISE EXCEPTION 'Minimum monthly dollar amount purchased parameter must be > $0.00';
    END IF;

    last_month_start := CURRENT_DATE - '3 month'::interval;
    last_month_start := to_date((extract(YEAR FROM last_month_start) || '-' || extract(MONTH FROM last_month_start) || '-01'),'YYYY-MM-DD');
    last_month_end := LAST_DAY(last_month_start);

    /*
    Create a temporary storage area for Customer IDs.
    */
    CREATE TEMPORARY TABLE tmpCustomer (customer_id INTEGER NOT NULL PRIMARY KEY);

    /*
    Find all customers meeting the monthly purchase requirements
    */

    tmpSQL := 'INSERT INTO tmpCustomer (customer_id)
        SELECT p.customer_id
        FROM payment AS p
        WHERE DATE(p.payment_date) BETWEEN '||quote_literal(last_month_start) ||' AND '|| quote_literal(last_month_end) || '
        GROUP BY customer_id
        HAVING SUM(p.amount) > '|| min_dollar_amount_purchased || '
        AND COUNT(customer_id) > ' ||min_monthly_purchases ;

    EXECUTE tmpSQL;

    /*
    Output ALL customer information of matching rewardees.
    Customize output as needed.
    */
    FOR rr IN EXECUTE 'SELECT c.* FROM tmpCustomer AS t INNER JOIN customer AS c ON t.customer_id = c.customer_id' LOOP
        RETURN NEXT rr;
    END LOOP;

    /* Clean up */
    tmpSQL := 'DROP TABLE tmpCustomer';
    EXECUTE tmpSQL;

RETURN;
END
$BODY$
  LANGUAGE plpgsql VOLATILE SECURITY DEFINER
  COST 100
  ROWS 1000;

-- ----------------------------
-- View structure for actor_info
-- ----------------------------
DROP VIEW IF EXISTS "public"."actor_info";
CREATE VIEW "public"."actor_info" AS  SELECT a.actor_id,
    a.first_name,
    a.last_name,
    group_concat(DISTINCT (c.name || ': '::text) || (( SELECT group_concat(f.title) AS group_concat
           FROM film f
             JOIN film_category fc_1 ON f.film_id = fc_1.film_id
             JOIN film_actor fa_1 ON f.film_id = fa_1.film_id
          WHERE fc_1.category_id = c.category_id AND fa_1.actor_id = a.actor_id
          GROUP BY fa_1.actor_id))) AS film_info
   FROM actor a
     LEFT JOIN film_actor fa ON a.actor_id = fa.actor_id
     LEFT JOIN film_category fc ON fa.film_id = fc.film_id
     LEFT JOIN category c ON fc.category_id = c.category_id
  GROUP BY a.actor_id, a.first_name, a.last_name;

-- ----------------------------
-- View structure for customer_list
-- ----------------------------
DROP VIEW IF EXISTS "public"."customer_list";
CREATE VIEW "public"."customer_list" AS  SELECT cu.customer_id AS id,
    (cu.first_name || ' '::text) || cu.last_name AS name,
    a.address,
    a.postal_code AS "zip code",
    a.phone,
    city.city,
    country.country,
        CASE
            WHEN cu.activebool THEN 'active'::text
            ELSE ''::text
        END AS notes,
    cu.store_id AS sid
   FROM customer cu
     JOIN address a ON cu.address_id = a.address_id
     JOIN city ON a.city_id = city.city_id
     JOIN country ON city.country_id = country.country_id;

-- ----------------------------
-- View structure for film_list
-- ----------------------------
DROP VIEW IF EXISTS "public"."film_list";
CREATE VIEW "public"."film_list" AS  SELECT film.film_id AS fid,
    film.title,
    film.description,
    category.name AS category,
    film.rental_rate AS price,
    film.length,
    film.rating,
    group_concat((actor.first_name || ' '::text) || actor.last_name) AS actors
   FROM category
     LEFT JOIN film_category ON category.category_id = film_category.category_id
     LEFT JOIN film ON film_category.film_id = film.film_id
     JOIN film_actor ON film.film_id = film_actor.film_id
     JOIN actor ON film_actor.actor_id = actor.actor_id
  GROUP BY film.film_id, film.title, film.description, category.name, film.rental_rate, film.length, film.rating;

-- ----------------------------
-- View structure for nicer_but_slower_film_list
-- ----------------------------
DROP VIEW IF EXISTS "public"."nicer_but_slower_film_list";
CREATE VIEW "public"."nicer_but_slower_film_list" AS  SELECT film.film_id AS fid,
    film.title,
    film.description,
    category.name AS category,
    film.rental_rate AS price,
    film.length,
    film.rating,
    group_concat(((upper("substring"(actor.first_name, 1, 1)) || lower("substring"(actor.first_name, 2))) || upper("substring"(actor.last_name, 1, 1))) || lower("substring"(actor.last_name, 2))) AS actors
   FROM category
     LEFT JOIN film_category ON category.category_id = film_category.category_id
     LEFT JOIN film ON film_category.film_id = film.film_id
     JOIN film_actor ON film.film_id = film_actor.film_id
     JOIN actor ON film_actor.actor_id = actor.actor_id
  GROUP BY film.film_id, film.title, film.description, category.name, film.rental_rate, film.length, film.rating;

-- ----------------------------
-- View structure for sales_by_film_category
-- ----------------------------
DROP VIEW IF EXISTS "public"."sales_by_film_category";
CREATE VIEW "public"."sales_by_film_category" AS  SELECT c.name AS category,
    sum(p.amount) AS total_sales
   FROM payment p
     JOIN rental r ON p.rental_id = r.rental_id
     JOIN inventory i ON r.inventory_id = i.inventory_id
     JOIN film f ON i.film_id = f.film_id
     JOIN film_category fc ON f.film_id = fc.film_id
     JOIN category c ON fc.category_id = c.category_id
  GROUP BY c.name
  ORDER BY (sum(p.amount)) DESC;

-- ----------------------------
-- View structure for sales_by_store
-- ----------------------------
DROP VIEW IF EXISTS "public"."sales_by_store";
CREATE VIEW "public"."sales_by_store" AS  SELECT (c.city || ','::text) || cy.country AS store,
    (m.first_name || ' '::text) || m.last_name AS manager,
    sum(p.amount) AS total_sales
   FROM payment p
     JOIN rental r ON p.rental_id = r.rental_id
     JOIN inventory i ON r.inventory_id = i.inventory_id
     JOIN store s ON i.store_id = s.store_id
     JOIN address a ON s.address_id = a.address_id
     JOIN city c ON a.city_id = c.city_id
     JOIN country cy ON c.country_id = cy.country_id
     JOIN staff m ON s.manager_staff_id = m.staff_id
  GROUP BY cy.country, c.city, s.store_id, m.first_name, m.last_name
  ORDER BY cy.country, c.city;

-- ----------------------------
-- View structure for staff_list
-- ----------------------------
DROP VIEW IF EXISTS "public"."staff_list";
CREATE VIEW "public"."staff_list" AS  SELECT s.staff_id AS id,
    (s.first_name || ' '::text) || s.last_name AS name,
    a.address,
    a.postal_code AS "zip code",
    a.phone,
    city.city,
    country.country,
    s.store_id AS sid
   FROM staff s
     JOIN address a ON s.address_id = a.address_id
     JOIN city ON a.city_id = city.city_id
     JOIN country ON city.country_id = country.country_id;

-- ----------------------------
-- Alter sequences owned by
-- ----------------------------
SELECT setval('"public"."actor_actor_id_seq"', 200, true);

-- ----------------------------
-- Alter sequences owned by
-- ----------------------------
SELECT setval('"public"."address_address_id_seq"', 605, true);

-- ----------------------------
-- Alter sequences owned by
-- ----------------------------
SELECT setval('"public"."category_category_id_seq"', 16, true);

-- ----------------------------
-- Alter sequences owned by
-- ----------------------------
SELECT setval('"public"."city_city_id_seq"', 600, true);

-- ----------------------------
-- Alter sequences owned by
-- ----------------------------
SELECT setval('"public"."country_country_id_seq"', 109, true);

-- ----------------------------
-- Alter sequences owned by
-- ----------------------------
SELECT setval('"public"."customer_customer_id_seq"', 599, true);

-- ----------------------------
-- Alter sequences owned by
-- ----------------------------
SELECT setval('"public"."film_film_id_seq"', 1000, true);

-- ----------------------------
-- Alter sequences owned by
-- ----------------------------
SELECT setval('"public"."inventory_inventory_id_seq"', 4581, true);

-- ----------------------------
-- Alter sequences owned by
-- ----------------------------
SELECT setval('"public"."language_language_id_seq"', 6, true);

-- ----------------------------
-- Alter sequences owned by
-- ----------------------------
SELECT setval('"public"."payment_payment_id_seq"', 32098, true);

-- ----------------------------
-- Alter sequences owned by
-- ----------------------------
SELECT setval('"public"."rental_rental_id_seq"', 16049, true);

-- ----------------------------
-- Alter sequences owned by
-- ----------------------------
SELECT setval('"public"."staff_staff_id_seq"', 2, true);

-- ----------------------------
-- Alter sequences owned by
-- ----------------------------
SELECT setval('"public"."store_store_id_seq"', 2, true);

-- ----------------------------
-- Indexes structure for table actor
-- ----------------------------
CREATE INDEX "idx_actor_last_name" ON "public"."actor" USING btree (
  "last_name" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);

-- ----------------------------
-- Triggers structure for table actor
-- ----------------------------
CREATE TRIGGER "last_updated" BEFORE UPDATE ON "public"."actor"
FOR EACH ROW
EXECUTE PROCEDURE "public"."last_updated"();

-- ----------------------------
-- Primary Key structure for table actor
-- ----------------------------
ALTER TABLE "public"."actor" ADD CONSTRAINT "actor_pkey" PRIMARY KEY ("actor_id");

-- ----------------------------
-- Indexes structure for table address
-- ----------------------------
CREATE INDEX "idx_fk_city_id" ON "public"."address" USING btree (
  "city_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Triggers structure for table address
-- ----------------------------
CREATE TRIGGER "last_updated" BEFORE UPDATE ON "public"."address"
FOR EACH ROW
EXECUTE PROCEDURE "public"."last_updated"();

-- ----------------------------
-- Primary Key structure for table address
-- ----------------------------
ALTER TABLE "public"."address" ADD CONSTRAINT "address_pkey" PRIMARY KEY ("address_id");

-- ----------------------------
-- Triggers structure for table category
-- ----------------------------
CREATE TRIGGER "last_updated" BEFORE UPDATE ON "public"."category"
FOR EACH ROW
EXECUTE PROCEDURE "public"."last_updated"();

-- ----------------------------
-- Primary Key structure for table category
-- ----------------------------
ALTER TABLE "public"."category" ADD CONSTRAINT "category_pkey" PRIMARY KEY ("category_id");

-- ----------------------------
-- Indexes structure for table city
-- ----------------------------
CREATE INDEX "idx_fk_country_id" ON "public"."city" USING btree (
  "country_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Triggers structure for table city
-- ----------------------------
CREATE TRIGGER "last_updated" BEFORE UPDATE ON "public"."city"
FOR EACH ROW
EXECUTE PROCEDURE "public"."last_updated"();

-- ----------------------------
-- Primary Key structure for table city
-- ----------------------------
ALTER TABLE "public"."city" ADD CONSTRAINT "city_pkey" PRIMARY KEY ("city_id");

-- ----------------------------
-- Triggers structure for table country
-- ----------------------------
CREATE TRIGGER "last_updated" BEFORE UPDATE ON "public"."country"
FOR EACH ROW
EXECUTE PROCEDURE "public"."last_updated"();

-- ----------------------------
-- Primary Key structure for table country
-- ----------------------------
ALTER TABLE "public"."country" ADD CONSTRAINT "country_pkey" PRIMARY KEY ("country_id");

-- ----------------------------
-- Indexes structure for table customer
-- ----------------------------
CREATE INDEX "idx_fk_address_id" ON "public"."customer" USING btree (
  "address_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "idx_fk_store_id" ON "public"."customer" USING btree (
  "store_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "idx_last_name" ON "public"."customer" USING btree (
  "last_name" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);

-- ----------------------------
-- Triggers structure for table customer
-- ----------------------------
CREATE TRIGGER "last_updated" BEFORE UPDATE ON "public"."customer"
FOR EACH ROW
EXECUTE PROCEDURE "public"."last_updated"();

-- ----------------------------
-- Primary Key structure for table customer
-- ----------------------------
ALTER TABLE "public"."customer" ADD CONSTRAINT "customer_pkey" PRIMARY KEY ("customer_id");

-- ----------------------------
-- Indexes structure for table film
-- ----------------------------
CREATE INDEX "film_fulltext_idx" ON "public"."film" USING gist (
  "fulltext" "pg_catalog"."tsvector_ops"
);
CREATE INDEX "idx_fk_language_id" ON "public"."film" USING btree (
  "language_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "idx_fk_original_language_id" ON "public"."film" USING btree (
  "original_language_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "idx_title" ON "public"."film" USING btree (
  "title" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);

-- ----------------------------
-- Triggers structure for table film
-- ----------------------------
CREATE TRIGGER "film_fulltext_trigger" BEFORE INSERT OR UPDATE ON "public"."film"
FOR EACH ROW
EXECUTE PROCEDURE "pg_catalog"."tsvector_update_trigger"('fulltext', 'pg_catalog.english', 'title', 'description');
CREATE TRIGGER "last_updated" BEFORE UPDATE ON "public"."film"
FOR EACH ROW
EXECUTE PROCEDURE "public"."last_updated"();

-- ----------------------------
-- Primary Key structure for table film
-- ----------------------------
ALTER TABLE "public"."film" ADD CONSTRAINT "film_pkey" PRIMARY KEY ("film_id");

-- ----------------------------
-- Indexes structure for table film_actor
-- ----------------------------
CREATE INDEX "idx_fk_film_id" ON "public"."film_actor" USING btree (
  "film_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Triggers structure for table film_actor
-- ----------------------------
CREATE TRIGGER "last_updated" BEFORE UPDATE ON "public"."film_actor"
FOR EACH ROW
EXECUTE PROCEDURE "public"."last_updated"();

-- ----------------------------
-- Primary Key structure for table film_actor
-- ----------------------------
ALTER TABLE "public"."film_actor" ADD CONSTRAINT "film_actor_pkey" PRIMARY KEY ("actor_id", "film_id");

-- ----------------------------
-- Triggers structure for table film_category
-- ----------------------------
CREATE TRIGGER "last_updated" BEFORE UPDATE ON "public"."film_category"
FOR EACH ROW
EXECUTE PROCEDURE "public"."last_updated"();

-- ----------------------------
-- Primary Key structure for table film_category
-- ----------------------------
ALTER TABLE "public"."film_category" ADD CONSTRAINT "film_category_pkey" PRIMARY KEY ("film_id", "category_id");

-- ----------------------------
-- Indexes structure for table inventory
-- ----------------------------
CREATE INDEX "idx_store_id_film_id" ON "public"."inventory" USING btree (
  "store_id" "pg_catalog"."int4_ops" ASC NULLS LAST,
  "film_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Triggers structure for table inventory
-- ----------------------------
CREATE TRIGGER "last_updated" BEFORE UPDATE ON "public"."inventory"
FOR EACH ROW
EXECUTE PROCEDURE "public"."last_updated"();

-- ----------------------------
-- Primary Key structure for table inventory
-- ----------------------------
ALTER TABLE "public"."inventory" ADD CONSTRAINT "inventory_pkey" PRIMARY KEY ("inventory_id");

-- ----------------------------
-- Triggers structure for table language
-- ----------------------------
CREATE TRIGGER "last_updated" BEFORE UPDATE ON "public"."language"
FOR EACH ROW
EXECUTE PROCEDURE "public"."last_updated"();

-- ----------------------------
-- Primary Key structure for table language
-- ----------------------------
ALTER TABLE "public"."language" ADD CONSTRAINT "language_pkey" PRIMARY KEY ("language_id");

-- ----------------------------
-- Indexes structure for table payment_p2022_01
-- ----------------------------
CREATE INDEX "idx_fk_payment_p2022_01_customer_id" ON "public"."payment_p2022_01" USING btree (
  "customer_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "idx_fk_payment_p2022_01_staff_id" ON "public"."payment_p2022_01" USING btree (
  "staff_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "payment_p2022_01_customer_id_idx" ON "public"."payment_p2022_01" USING btree (
  "customer_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table payment_p2022_01
-- ----------------------------
ALTER TABLE "public"."payment_p2022_01" ADD CONSTRAINT "payment_p2022_01_pkey" PRIMARY KEY ("payment_date", "payment_id");

-- ----------------------------
-- Indexes structure for table payment_p2022_02
-- ----------------------------
CREATE INDEX "idx_fk_payment_p2022_02_customer_id" ON "public"."payment_p2022_02" USING btree (
  "customer_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "idx_fk_payment_p2022_02_staff_id" ON "public"."payment_p2022_02" USING btree (
  "staff_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "payment_p2022_02_customer_id_idx" ON "public"."payment_p2022_02" USING btree (
  "customer_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table payment_p2022_02
-- ----------------------------
ALTER TABLE "public"."payment_p2022_02" ADD CONSTRAINT "payment_p2022_02_pkey" PRIMARY KEY ("payment_date", "payment_id");

-- ----------------------------
-- Indexes structure for table payment_p2022_03
-- ----------------------------
CREATE INDEX "idx_fk_payment_p2022_03_customer_id" ON "public"."payment_p2022_03" USING btree (
  "customer_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "idx_fk_payment_p2022_03_staff_id" ON "public"."payment_p2022_03" USING btree (
  "staff_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "payment_p2022_03_customer_id_idx" ON "public"."payment_p2022_03" USING btree (
  "customer_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table payment_p2022_03
-- ----------------------------
ALTER TABLE "public"."payment_p2022_03" ADD CONSTRAINT "payment_p2022_03_pkey" PRIMARY KEY ("payment_date", "payment_id");

-- ----------------------------
-- Indexes structure for table payment_p2022_04
-- ----------------------------
CREATE INDEX "idx_fk_payment_p2022_04_customer_id" ON "public"."payment_p2022_04" USING btree (
  "customer_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "idx_fk_payment_p2022_04_staff_id" ON "public"."payment_p2022_04" USING btree (
  "staff_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "payment_p2022_04_customer_id_idx" ON "public"."payment_p2022_04" USING btree (
  "customer_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table payment_p2022_04
-- ----------------------------
ALTER TABLE "public"."payment_p2022_04" ADD CONSTRAINT "payment_p2022_04_pkey" PRIMARY KEY ("payment_date", "payment_id");

-- ----------------------------
-- Indexes structure for table payment_p2022_05
-- ----------------------------
CREATE INDEX "idx_fk_payment_p2022_05_customer_id" ON "public"."payment_p2022_05" USING btree (
  "customer_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "idx_fk_payment_p2022_05_staff_id" ON "public"."payment_p2022_05" USING btree (
  "staff_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "payment_p2022_05_customer_id_idx" ON "public"."payment_p2022_05" USING btree (
  "customer_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table payment_p2022_05
-- ----------------------------
ALTER TABLE "public"."payment_p2022_05" ADD CONSTRAINT "payment_p2022_05_pkey" PRIMARY KEY ("payment_date", "payment_id");

-- ----------------------------
-- Indexes structure for table payment_p2022_06
-- ----------------------------
CREATE INDEX "idx_fk_payment_p2022_06_customer_id" ON "public"."payment_p2022_06" USING btree (
  "customer_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "idx_fk_payment_p2022_06_staff_id" ON "public"."payment_p2022_06" USING btree (
  "staff_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "payment_p2022_06_customer_id_idx" ON "public"."payment_p2022_06" USING btree (
  "customer_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table payment_p2022_06
-- ----------------------------
ALTER TABLE "public"."payment_p2022_06" ADD CONSTRAINT "payment_p2022_06_pkey" PRIMARY KEY ("payment_date", "payment_id");

-- ----------------------------
-- Primary Key structure for table payment_p2022_07
-- ----------------------------
ALTER TABLE "public"."payment_p2022_07" ADD CONSTRAINT "payment_p2022_07_pkey" PRIMARY KEY ("payment_date", "payment_id");

-- ----------------------------
-- Indexes structure for table rental
-- ----------------------------
CREATE INDEX "idx_fk_inventory_id" ON "public"."rental" USING btree (
  "inventory_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE UNIQUE INDEX "idx_unq_rental_rental_date_inventory_id_customer_id" ON "public"."rental" USING btree (
  "rental_date" "pg_catalog"."timestamptz_ops" ASC NULLS LAST,
  "inventory_id" "pg_catalog"."int4_ops" ASC NULLS LAST,
  "customer_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Triggers structure for table rental
-- ----------------------------
CREATE TRIGGER "last_updated" BEFORE UPDATE ON "public"."rental"
FOR EACH ROW
EXECUTE PROCEDURE "public"."last_updated"();

-- ----------------------------
-- Primary Key structure for table rental
-- ----------------------------
ALTER TABLE "public"."rental" ADD CONSTRAINT "rental_pkey" PRIMARY KEY ("rental_id");

-- ----------------------------
-- Triggers structure for table staff
-- ----------------------------
CREATE TRIGGER "last_updated" BEFORE UPDATE ON "public"."staff"
FOR EACH ROW
EXECUTE PROCEDURE "public"."last_updated"();

-- ----------------------------
-- Primary Key structure for table staff
-- ----------------------------
ALTER TABLE "public"."staff" ADD CONSTRAINT "staff_pkey" PRIMARY KEY ("staff_id");

-- ----------------------------
-- Indexes structure for table store
-- ----------------------------
CREATE UNIQUE INDEX "idx_unq_manager_staff_id" ON "public"."store" USING btree (
  "manager_staff_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Triggers structure for table store
-- ----------------------------
CREATE TRIGGER "last_updated" BEFORE UPDATE ON "public"."store"
FOR EACH ROW
EXECUTE PROCEDURE "public"."last_updated"();

-- ----------------------------
-- Primary Key structure for table store
-- ----------------------------
ALTER TABLE "public"."store" ADD CONSTRAINT "store_pkey" PRIMARY KEY ("store_id");

-- ----------------------------
-- Primary Key structure for table payment
-- ----------------------------
ALTER TABLE "public"."payment" ADD CONSTRAINT "payment_pkey" PRIMARY KEY ("payment_date", "payment_id");

-- ----------------------------
-- Foreign Keys structure for table address
-- ----------------------------
ALTER TABLE "public"."address" ADD CONSTRAINT "address_city_id_fkey" FOREIGN KEY ("city_id") REFERENCES "public"."city" ("city_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- ----------------------------
-- Foreign Keys structure for table city
-- ----------------------------
ALTER TABLE "public"."city" ADD CONSTRAINT "city_country_id_fkey" FOREIGN KEY ("country_id") REFERENCES "public"."country" ("country_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- ----------------------------
-- Foreign Keys structure for table customer
-- ----------------------------
ALTER TABLE "public"."customer" ADD CONSTRAINT "customer_address_id_fkey" FOREIGN KEY ("address_id") REFERENCES "public"."address" ("address_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "public"."customer" ADD CONSTRAINT "customer_store_id_fkey" FOREIGN KEY ("store_id") REFERENCES "public"."store" ("store_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- ----------------------------
-- Foreign Keys structure for table film
-- ----------------------------
ALTER TABLE "public"."film" ADD CONSTRAINT "film_language_id_fkey" FOREIGN KEY ("language_id") REFERENCES "public"."language" ("language_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "public"."film" ADD CONSTRAINT "film_original_language_id_fkey" FOREIGN KEY ("original_language_id") REFERENCES "public"."language" ("language_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- ----------------------------
-- Foreign Keys structure for table film_actor
-- ----------------------------
ALTER TABLE "public"."film_actor" ADD CONSTRAINT "film_actor_actor_id_fkey" FOREIGN KEY ("actor_id") REFERENCES "public"."actor" ("actor_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "public"."film_actor" ADD CONSTRAINT "film_actor_film_id_fkey" FOREIGN KEY ("film_id") REFERENCES "public"."film" ("film_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- ----------------------------
-- Foreign Keys structure for table film_category
-- ----------------------------
ALTER TABLE "public"."film_category" ADD CONSTRAINT "film_category_category_id_fkey" FOREIGN KEY ("category_id") REFERENCES "public"."category" ("category_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "public"."film_category" ADD CONSTRAINT "film_category_film_id_fkey" FOREIGN KEY ("film_id") REFERENCES "public"."film" ("film_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- ----------------------------
-- Foreign Keys structure for table inventory
-- ----------------------------
ALTER TABLE "public"."inventory" ADD CONSTRAINT "inventory_film_id_fkey" FOREIGN KEY ("film_id") REFERENCES "public"."film" ("film_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "public"."inventory" ADD CONSTRAINT "inventory_store_id_fkey" FOREIGN KEY ("store_id") REFERENCES "public"."store" ("store_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- ----------------------------
-- Foreign Keys structure for table payment_p2022_01
-- ----------------------------
ALTER TABLE "public"."payment_p2022_01" ADD CONSTRAINT "payment_p2022_01_customer_id_fkey" FOREIGN KEY ("customer_id") REFERENCES "public"."customer" ("customer_id") ON DELETE NO ACTION ON UPDATE NO ACTION;
ALTER TABLE "public"."payment_p2022_01" ADD CONSTRAINT "payment_p2022_01_rental_id_fkey" FOREIGN KEY ("rental_id") REFERENCES "public"."rental" ("rental_id") ON DELETE NO ACTION ON UPDATE NO ACTION;
ALTER TABLE "public"."payment_p2022_01" ADD CONSTRAINT "payment_p2022_01_staff_id_fkey" FOREIGN KEY ("staff_id") REFERENCES "public"."staff" ("staff_id") ON DELETE NO ACTION ON UPDATE NO ACTION;

-- ----------------------------
-- Foreign Keys structure for table payment_p2022_02
-- ----------------------------
ALTER TABLE "public"."payment_p2022_02" ADD CONSTRAINT "payment_p2022_02_customer_id_fkey" FOREIGN KEY ("customer_id") REFERENCES "public"."customer" ("customer_id") ON DELETE NO ACTION ON UPDATE NO ACTION;
ALTER TABLE "public"."payment_p2022_02" ADD CONSTRAINT "payment_p2022_02_rental_id_fkey" FOREIGN KEY ("rental_id") REFERENCES "public"."rental" ("rental_id") ON DELETE NO ACTION ON UPDATE NO ACTION;
ALTER TABLE "public"."payment_p2022_02" ADD CONSTRAINT "payment_p2022_02_staff_id_fkey" FOREIGN KEY ("staff_id") REFERENCES "public"."staff" ("staff_id") ON DELETE NO ACTION ON UPDATE NO ACTION;

-- ----------------------------
-- Foreign Keys structure for table payment_p2022_03
-- ----------------------------
ALTER TABLE "public"."payment_p2022_03" ADD CONSTRAINT "payment_p2022_03_customer_id_fkey" FOREIGN KEY ("customer_id") REFERENCES "public"."customer" ("customer_id") ON DELETE NO ACTION ON UPDATE NO ACTION;
ALTER TABLE "public"."payment_p2022_03" ADD CONSTRAINT "payment_p2022_03_rental_id_fkey" FOREIGN KEY ("rental_id") REFERENCES "public"."rental" ("rental_id") ON DELETE NO ACTION ON UPDATE NO ACTION;
ALTER TABLE "public"."payment_p2022_03" ADD CONSTRAINT "payment_p2022_03_staff_id_fkey" FOREIGN KEY ("staff_id") REFERENCES "public"."staff" ("staff_id") ON DELETE NO ACTION ON UPDATE NO ACTION;

-- ----------------------------
-- Foreign Keys structure for table payment_p2022_04
-- ----------------------------
ALTER TABLE "public"."payment_p2022_04" ADD CONSTRAINT "payment_p2022_04_customer_id_fkey" FOREIGN KEY ("customer_id") REFERENCES "public"."customer" ("customer_id") ON DELETE NO ACTION ON UPDATE NO ACTION;
ALTER TABLE "public"."payment_p2022_04" ADD CONSTRAINT "payment_p2022_04_rental_id_fkey" FOREIGN KEY ("rental_id") REFERENCES "public"."rental" ("rental_id") ON DELETE NO ACTION ON UPDATE NO ACTION;
ALTER TABLE "public"."payment_p2022_04" ADD CONSTRAINT "payment_p2022_04_staff_id_fkey" FOREIGN KEY ("staff_id") REFERENCES "public"."staff" ("staff_id") ON DELETE NO ACTION ON UPDATE NO ACTION;

-- ----------------------------
-- Foreign Keys structure for table payment_p2022_05
-- ----------------------------
ALTER TABLE "public"."payment_p2022_05" ADD CONSTRAINT "payment_p2022_05_customer_id_fkey" FOREIGN KEY ("customer_id") REFERENCES "public"."customer" ("customer_id") ON DELETE NO ACTION ON UPDATE NO ACTION;
ALTER TABLE "public"."payment_p2022_05" ADD CONSTRAINT "payment_p2022_05_rental_id_fkey" FOREIGN KEY ("rental_id") REFERENCES "public"."rental" ("rental_id") ON DELETE NO ACTION ON UPDATE NO ACTION;
ALTER TABLE "public"."payment_p2022_05" ADD CONSTRAINT "payment_p2022_05_staff_id_fkey" FOREIGN KEY ("staff_id") REFERENCES "public"."staff" ("staff_id") ON DELETE NO ACTION ON UPDATE NO ACTION;

-- ----------------------------
-- Foreign Keys structure for table payment_p2022_06
-- ----------------------------
ALTER TABLE "public"."payment_p2022_06" ADD CONSTRAINT "payment_p2022_06_customer_id_fkey" FOREIGN KEY ("customer_id") REFERENCES "public"."customer" ("customer_id") ON DELETE NO ACTION ON UPDATE NO ACTION;
ALTER TABLE "public"."payment_p2022_06" ADD CONSTRAINT "payment_p2022_06_rental_id_fkey" FOREIGN KEY ("rental_id") REFERENCES "public"."rental" ("rental_id") ON DELETE NO ACTION ON UPDATE NO ACTION;
ALTER TABLE "public"."payment_p2022_06" ADD CONSTRAINT "payment_p2022_06_staff_id_fkey" FOREIGN KEY ("staff_id") REFERENCES "public"."staff" ("staff_id") ON DELETE NO ACTION ON UPDATE NO ACTION;

-- ----------------------------
-- Foreign Keys structure for table rental
-- ----------------------------
ALTER TABLE "public"."rental" ADD CONSTRAINT "rental_customer_id_fkey" FOREIGN KEY ("customer_id") REFERENCES "public"."customer" ("customer_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "public"."rental" ADD CONSTRAINT "rental_inventory_id_fkey" FOREIGN KEY ("inventory_id") REFERENCES "public"."inventory" ("inventory_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "public"."rental" ADD CONSTRAINT "rental_staff_id_fkey" FOREIGN KEY ("staff_id") REFERENCES "public"."staff" ("staff_id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- ----------------------------
-- Foreign Keys structure for table staff
-- ----------------------------
ALTER TABLE "public"."staff" ADD CONSTRAINT "staff_address_id_fkey" FOREIGN KEY ("address_id") REFERENCES "public"."address" ("address_id") ON DELETE RESTRICT ON UPDATE CASCADE;
ALTER TABLE "public"."staff" ADD CONSTRAINT "staff_store_id_fkey" FOREIGN KEY ("store_id") REFERENCES "public"."store" ("store_id") ON DELETE NO ACTION ON UPDATE NO ACTION;

-- ----------------------------
-- Foreign Keys structure for table store
-- ----------------------------
ALTER TABLE "public"."store" ADD CONSTRAINT "store_address_id_fkey" FOREIGN KEY ("address_id") REFERENCES "public"."address" ("address_id") ON DELETE RESTRICT ON UPDATE CASCADE;
