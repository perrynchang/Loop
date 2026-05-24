"""
Task Capo: Knowledge Capacity
Synthetic biographies (bioS dataset) for measuring bits-per-parameter knowledge storage.
Each biography has 6 attributes; models are trained for 100 exposures each.

Attribute pools and sentence templates match the author's reference implementation exactly
(fields/ directory in allen4.1code/data-synthetic-pretrain/Capo-bioS-bioR/).
Tokenization uses GPT-2 (vocab=50257) to match the paper's bits-per-parameter metric.
"""
import random
import torch
from torch.utils.data import IterableDataset

# ── Attribute pools (from author's fields/ directory) ─────────────────────────

FIRST_NAMES = [
    "Jacob","Michael","Joshua","Matthew","Daniel","Christopher","Andrew","Ethan","Joseph","William",
    "Anthony","David","Alexander","Nicholas","Ryan","Tyler","James","John","Jonathan","Noah",
    "Brandon","Christian","Dylan","Samuel","Benjamin","Nathan","Zachary","Logan","Justin","Gabriel",
    "Jose","Austin","Kevin","Elijah","Caleb","Robert","Thomas","Jordan","Cameron","Jack",
    "Hunter","Jackson","Angel","Isaiah","Evan","Isaac","Luke","Mason","Jayden","Jason",
    "Gavin","Aaron","Connor","Aiden","Aidan","Kyle","Juan","Charles","Luis","Adam",
    "Lucas","Brian","Eric","Adrian","Nathaniel","Sean","Alex","Carlos","Bryan","Ian",
    "Owen","Jesus","Landon","Julian","Chase","Cole","Diego","Jeremiah","Steven","Sebastian",
    "Xavier","Timothy","Carter","Wyatt","Brayden","Blake","Hayden","Devin","Cody","Richard",
    "Seth","Dominic","Jaden","Antonio","Miguel","Liam","Patrick","Carson","Jesse","Tristan",
    "Alejandro","Henry","Victor","Trevor","Bryce","Jake","Riley","Colin","Jared","Jeremy",
    "Mark","Caden","Garrett","Parker","Marcus","Vincent","Kaleb","Kaden","Brady","Colton",
    "Kenneth","Joel","Oscar","Josiah","Jorge","Ashton","Cooper","Tanner","Eduardo","Paul",
    "Edward","Ivan","Preston","Maxwell","Alan","Levi","Stephen","Grant","Nicolas","Dakota",
    "Omar","Alexis","George","Eli","Collin","Spencer","Gage","Max","Ricardo","Cristian",
    "Derek","Micah","Brody","Francisco","Nolan","Ayden","Dalton","Shane","Peter","Damian",
    "Jeffrey","Brendan","Travis","Fernando","Peyton","Conner","Andres","Javier","Giovanni","Shawn",
    "Braden","Jonah","Bradley","Cesar","Emmanuel","Manuel","Edgar","Mario","Erik","Edwin",
    "Johnathan","Devon","Erick","Wesley","Oliver","Trenton","Hector","Malachi","Jalen","Raymond",
    "Gregory","Abraham","Elias","Leonardo","Sergio","Donovan","Colby","Marco","Bryson","Martin",
    "Emily","Madison","Emma","Olivia","Hannah","Abigail","Isabella","Samantha","Elizabeth","Ashley",
    "Sarah","Sophia","Alyssa","Grace","Ava","Taylor","Brianna","Lauren","Chloe","Natalie",
    "Kayla","Jessica","Anna","Victoria","Mia","Hailey","Sydney","Jasmine","Julia","Morgan",
    "Destiny","Rachel","Ella","Kaitlyn","Megan","Katherine","Savannah","Jennifer","Alexandra","Allison",
    "Haley","Maria","Kaylee","Lily","Makayla","Brooke","Nicole","Mackenzie","Addison","Stephanie",
    "Lillian","Andrea","Faith","Zoe","Kimberly","Madeline","Alexa","Katelyn","Gabriella","Gabrielle",
    "Trinity","Amanda","Kylie","Mary","Paige","Leah","Jenna","Sara","Rebecca","Michelle",
    "Sofia","Vanessa","Angelina","Caroline","Avery","Audrey","Evelyn","Maya","Claire","Autumn",
    "Jocelyn","Ariana","Nevaeh","Arianna","Jada","Bailey","Brooklyn","Aaliyah","Amber","Isabel",
    "Mariah","Danielle","Melanie","Sierra","Erin","Amelia","Molly","Isabelle","Madelyn","Melissa",
    "Jacqueline","Marissa","Angela","Shelby","Leslie","Katie","Jade","Catherine","Diana","Aubrey",
    "Mya","Amy","Briana","Sophie","Gabriela","Breanna","Gianna","Kennedy","Gracie","Adriana",
    "Christina","Courtney","Daniela","Lydia","Kathryn","Valeria","Layla","Alexandria","Natalia","Laura",
    "Charlotte","Margaret","Cheyenne","Miranda","Mikayla","Naomi","Kelsey","Payton","Ana","Alicia",
    "Jillian","Daisy","Mckenzie","Ashlyn","Sabrina","Caitlin","Summer","Ruby","Valerie","Rylee",
    "Skylar","Lindsey","Kelly","Genesis","Zoey","Eva","Sadie","Alexia","Cassidy","Kylee",
    "Kendall","Jordyn","Kate","Jayla","Karen","Tiffany","Cassandra","Juliana","Reagan","Caitlyn",
    "Giselle","Serenity","Alondra","Lucy","Bianca","Kiara","Crystal","Erica","Angelica","Hope",
    "Chelsea","Alana","Liliana","Brittany","Camila","Makenzie","Lilly","Veronica","Abby","Jazmin",
    "Adrianna","Delaney","Karina","Ellie","Jasmin","Arthur","Rose","Reuben","Finn","Ivy",
]

MIDDLE_NAMES = [
    "Noah","Liam","Jacob","Mason","William","Ethan","Michael","Alexander","James","Elijah",
    "Daniel","Benjamin","Aiden","Jayden","Logan","Matthew","David","Joseph","Lucas","Jackson",
    "Anthony","Joshua","Samuel","Andrew","Gabriel","Christopher","John","Dylan","Carter","Isaac",
    "Ryan","Luke","Oliver","Nathan","Henry","Owen","Caleb","Wyatt","Christian","Sebastian",
    "Jack","Jonathan","Landon","Julian","Isaiah","Hunter","Levi","Aaron","Eli","Charles",
    "Thomas","Connor","Brayden","Nicholas","Jaxon","Jeremiah","Cameron","Evan","Adrian","Jordan",
    "Gavin","Grayson","Angel","Robert","Tyler","Josiah","Austin","Colton","Brandon","Jose",
    "Dominic","Kevin","Zachary","Ian","Chase","Jason","Adam","Ayden","Parker","Hudson",
    "Cooper","Nolan","Lincoln","Xavier","Carson","Jace","Justin","Easton","Mateo","Asher",
    "Bentley","Blake","Nathaniel","Jaxson","Leo","Kayden","Tristan","Luis","Elias","Brody",
    "Bryson","Juan","Vincent","Cole","Micah","Ryder","Theodore","Carlos","Ezra","Damian",
    "Miles","Santiago","Max","Jesus","Leonardo","Sawyer","Diego","Alex","Roman","Maxwell",
    "Eric","Greyson","Hayden","Giovanni","Wesley","Axel","Camden","Braxton","Ivan","Ashton",
    "Declan","Bryce","Timothy","Antonio","Silas","Kaiden","Ezekiel","Jonah","Weston","George",
    "Harrison","Steven","Miguel","Richard","Bryan","Kaleb","Victor","Aidan","Jameson","Joel",
    "Patrick","Jaden","Colin","Everett","Preston","Maddox","Edward","Alejandro","Kaden","Jesse",
    "Emmanuel","Kyle","Brian","Emmett","Jude","Marcus","Kingston","Kai","Alan","Malachi",
    "Grant","Jeremy","Riley","Jayce","Bennett","Abel","Ryker","Caden","Brantley","Luca",
    "Brady","Calvin","Sean","Oscar","Jake","Maverick","Abraham","Mark","Tucker","Nicolas",
    "Bradley","Kenneth","Avery","Cayden","King","Paul","Amir","Gael","Graham","Maximus",
    "Emma","Sophia","Olivia","Isabella","Ava","Mia","Abigail","Emily","Madison","Charlotte",
    "Elizabeth","Amelia","Chloe","Ella","Evelyn","Sofia","Harper","Grace","Addison","Victoria",
    "Natalie","Lily","Aubrey","Lillian","Zoey","Hannah","Layla","Brooklyn","Samantha","Zoe",
    "Leah","Scarlett","Camila","Savannah","Anna","Audrey","Allison","Aria","Gabriella","Hailey",
    "Claire","Sarah","Aaliyah","Kaylee","Nevaeh","Penelope","Alexa","Arianna","Stella","Alexis",
    "Bella","Nora","Ellie","Ariana","Lucy","Mila","Peyton","Genesis","Alyssa","Taylor",
    "Violet","Maya","Caroline","Madelyn","Skylar","Serenity","Ashley","Brianna","Kennedy","Autumn",
    "Eleanor","Kylie","Sadie","Paisley","Julia","Mackenzie","Sophie","Naomi","Eva","Khloe",
    "Katherine","Gianna","Melanie","Aubree","Piper","Ruby","Lydia","Faith","Madeline","Alexandra",
    "Kayla","Hazel","Lauren","Annabelle","Jasmine","Aurora","Alice","Makayla","Sydney","Bailey",
    "Luna","Maria","Reagan","Morgan","Isabelle","Rylee","Kimberly","Andrea","London","Elena",
    "Jocelyn","Natalia","Trinity","Eliana","Vivian","Cora","Quinn","Liliana","Molly","Jade",
    "Clara","Valentina","Mary","Brielle","Hadley","Kinsley","Willow","Brooke","Lilly","Delilah",
    "Payton","Mariah","Paige","Jordyn","Nicole","Mya","Josephine","Isabel","Lyla","Adeline",
    "Destiny","Ivy","Emilia","Rachel","Angelina","Valeria","Kendall","Sara","Ximena","Isla",
    "Aliyah","Reese","Vanessa","Juliana","Mckenzie","Amy","Laila","Adalynn","Emery","Margaret",
    "Eden","Gabrielle","Kaitlyn","Ariel","Gracie","Brooklynn","Melody","Jessica","Valerie","Adalyn",
    "Adriana","Elise","Michelle","Rebecca","Daisy","Everly","Katelyn","Ryleigh","Catherine","Norah",
    "Alaina","Athena","Leilani","Londyn","Eliza","Jayla","Summer","Lila","Makenzie","Izabella",
    "Daniela","Stephanie","Julianna","Rose","Alana","Harmony","Jennifer","Blair","Seth","Tate",
]

LAST_NAMES = [
    "Smith","Jones","Taylor","Brown","Williams","Wilson","Johnson","Davies","Patel","Robinson",
    "Wright","Thompson","Evans","Walker","White","Roberts","Green","Hall","Thomas","Clarke",
    "Jackson","Wood","Harris","Edwards","Turner","Martin","Cooper","Hill","Ward","Hughes",
    "Moore","Clark","King","Harrison","Lewis","Baker","Lee","Allen","Morris","Khan",
    "Scott","Watson","Davis","Parker","James","Bennett","Young","Phillips","Richardson","Mitchell",
    "Bailey","Carter","Cook","Singh","Shaw","Bell","Collins","Morgan","Kelly","Begum",
    "Miller","Cox","Hussain","Marshall","Simpson","Price","Anderson","Adams","Wilkinson","Ali",
    "Ahmed","Foster","Ellis","Murphy","Chapman","Mason","Gray","Richards","Webb","Griffiths",
    "Hunt","Palmer","Campbell","Holmes","Mills","Rogers","Barnes","Knight","Matthews","Barker",
    "Powell","Stevens","Kaur","Fisher","Butler","Dixon","Russell","Harvey","Pearson","Graham",
    "Fletcher","Murray","Howard","Shah","Gibson","Gill","Fox","Stewart","Elliott","Lloyd",
    "Andrews","Ford","Owen","West","Saunders","Reynolds","Day","Walsh","Brooks","Atkinson",
    "Payne","Cole","Bradley","Spencer","Pearce","Burton","Lawrence","Dawson","Ball","Rose",
    "Booth","Grant","Wells","Watts","Hudson","Hart","Armstrong","Perry","Newman","Jenkins",
    "Hunter","Webster","Lowe","Francis","Page","Hayes","Carr","Marsh","Stone","Riley",
    "Woods","Gregory","Barrett","Berry","Dunn","Newton","Holland","Porter","Oliver","Ryan",
    "Reid","Williamson","Parsons","O'Brien","Bird","Robertson","Reed","Bates","Dean","Walton",
    "Hawkins","Cooke","Harding","Ross","Henderson","Kennedy","Gardner","Lane","Burns","Bishop",
    "Burgess","Shepherd","Nicholson","Freeman","Cross","Hamilton","Hodgson","Warren","Sutton","Harper",
    "Yates","Nicholls","Robson","Chambers","Hardy","Curtis","Moss","Long","Akhtar","Coleman",
    "McDonald","Sharp","Potter","Jordan","George","Osborne","Gilbert","May","Hammond","Gordon",
    "Stevenson","Hutchinson","Wheeler","Wallace","Rowe","Willis","Read","Johnston","Mann","Stephenson",
    "Miles","Barber","Arnold","Byrne","Griffin","Slater","Nelson","Frost","Austin","Hewitt",
    "Buckley","Baxter","McCarthy","Whitehead","Higgins","O'Connor","Lambert","Hopkins","Barton","Greenwood",
    "Burke","Blake","Clayton","O'Neill","Goodwin","Doyle","Woodward","Bond","Kemp","Holt",
    "Thomson","Nash","Banks","Lawson","Miah","Davidson","Middleton","Cunningham","Barnett","Jennings",
    "Heath","Walters","Poole","French","Parry","Bibi","Fowler","Watkins","Jarvis","Lynch",
    "Quinn","Sullivan","Stanley","Norman","Stephens","Hartley","Rahman","Alexander","Lucas","Morton",
    "Peters","Knowles","Dickinson","Douglas","Field","Morrison","Preston","Stokes","Simmons","Black",
    "Gallagher","Barlow","Briggs","Gibbs","Todd","Tucker","Townsend","Ferguson","Parkinson","Burrows",
    "Thornton","Hayward","Pritchard","Rhodes","Thorpe","Fuller","Holden","Baldwin","Reeves","Lamb",
    "Norris","Sanders","Tomlinson","MacDonald","Hancock","Kent","Dale","Ashton","Howe","Abbott",
    "Davison","Glover","Kirby","Carroll","Weston","Kay","Kirk","Whittaker","Birch","Morley",
    "Mistry","Daniels","Goddard","Bryant","Dobson","Savage","Davey","Perkins","Warner","Skinner",
    "Bartlett","Brookes","Cartwright","Iqbal","Archer","Fraser","Sanderson","Bradshaw","Atkins","Smart",
    "Bull","Rees","Bentley","Patterson","Bolton","Haynes","Wilkins","Mahmood","Law","Little",
    "Wade","Malik","Howell","Schofield","Sharma","Dodd","Houghton","Butcher","Crawford","Hicks",
    "Henry","Wall","Short","Giles","Duncan","Coates","Manning","Noble","Clements","Duffy",
    "Sykes","Gould","Brennan","Farrell","Vaughan","Waters","Sheppard","Gibbons","Finch","Winter",
    "Naylor","Franklin","Flynn","Garner","Steele","Dyer","Marsden","Hooper","Vincent","Mohammed",
    "Joyce","Horton","Sharpe","Hobbs","Pickering","Humphreys","Dennis","Kerr","Fleming","Hurst",
    "Coles","Leach","Pratt","Randall","Moran","Howarth","Connolly","Peacock","Sinclair","Herbert",
    "Swift","Carpenter","Chandler","Chadwick","Blackburn","Pollard","Norton","Hale","Browne","Pugh",
    "Hilton","Welch","Faulkner","Parkin","Hanson","Kumar","Lyons","Cameron","Turnbull","Collier",
    "Allan","Bryan","Benson","Doherty","Charlton","Wallis","Chamberlain","Myers","Tyler","Conway",
    "Nixon","Paul","Metcalfe","Whitehouse","O'Sullivan","Gardiner","Lord","Joseph","Jacobs","Rice",
    "Rowley","Bowen","North","FitzGerald","Godfrey","Holloway","Bray","Hope","Talbot","Gough",
    "Connor","Hyde","Farmer","Storey","Potts","Nolan","Bruce","John","Butt","Donnelly",
    "McKenzie","Hargreaves","Brady","Parkes","Hassan","Forster","Pope","Eaton","Sims","Rowland",
    "Craig","Hirst","Lees","McLean","Boyle","Greaves","Summers","Mellor","Wyatt","Rigby",
    "Daly","Owens","Power","Ingram","Simmonds","Fry","Wild","Uddin","Gale","Neal",
    "Vickers","Marriott","Bradbury","Humphries","Goodman","Waller","Wong","Charles","Cullen","Spence",
    "Best","Islam","Ratcliffe","Barry","Massey","Stubbs","Bullock","Carey","Beaumont","Boyd",
    "Groves","Chan","Sadler","Leonard","Terry","Rayner","Bateman","Ahmad","Hills","Bowden",
    "Weaver","Hodges","Pike","Clifford","Reeve","Paterson","MacKenzie","Dalton","FitzPatrick","Welsh",
    "Small","Guest","Wills","Rodgers","Webber","Thorne","Barnard","Underwood","Stacey","Sweeney",
    "Allison","Langley","McKenna","O'Donnell","Woodcock","Woolley","Kenny","Hogg","Prince","Drew",
    "Bi","Oakley","Beard","Harrington","Kendall","Firth","Lawton","Parr","Draper","Hobson",
    "Beckett","Lacey","McDermott","Casey","Horne","Bacon","Humphrey","Lancaster","Bourne","Neale",
    "Jeffery","Betts","Dyson","Mercer","Seymour","Bedford","Crook","Guy","Reilly","Brook",
    "Gee","Plant","Burnett","Lock","Bowman","Leigh","Wilkes","Croft","Wheatley","McMahon",
    "Hubbard","Ashworth","Drake","Nichols","Stuart","Salmon","Partridge","Proctor","Sutcliffe","Johns",
    "Prior","Moody","Clarkson","Woodhouse","Maguire","McGrath","Platt","Chowdhury","Corbett","Haigh",
    "Harwood","Lake","Emery","Street","Lindsay","Cotton","Baines","Marks","Haines","Brewer",
    "Crane","Park","Bevan","Latham","Hutton","Stafford","Lister","Sandhu","Stanton","Beck",
    "McCann","Rashid","Milner","Brett","Hull","Sewell","Haywood","Bush","Parmar","Cope",
    "Aldridge","Hood","Waite","Bowers","McKay","Smyth","Wakefield","Johnstone","Steel","Tate",
    "Dickson","Ray","Mead","Daniel","England","Maxwell","English","Head","Whiting","Whittle",
    "Andrew","Garrett","Keen","Whitfield","Dunne","Butterworth","Dutton","Senior","Stott","Goodall",
    "Cummings","Westwood","Wainwright","Britton","Swain","Stringer","Hickman","Needham","Cannon","McLaughlin",
    "Roe","Ridley","Sutherland","Searle","Lockwood","Love","Fenton","Mansfield","Foley","Atherton",
    "Davenport","Masters","Grainger","Hallam","Hatton","Callaghan","Ryder","Cohen","Chappell","Kershaw",
    "Armitage","Wilcox","Lovell","Whelan","Howes","Radford","Newell","Childs","Choudhury","Li",
    "Darby","Cousins","Clegg","Whitaker","Burt","Irving","Salter","Coulson","Mortimer","Ireland",
    "Buck","Bright","Forbes","Hodson","Blackwell","Denton","Bannister","Dodds","Adamson","Mather",
    "Edge","Bland","Crossley","Rimmer","Nicholas","Bradford","Jenkinson","Nunn","Golding","Wardle",
    "Wilde","Forrest","Roper","McLoughlin","Mohamed","Ellison","Slade","Healey","Church","Kane",
    "Tanner","Kavanagh","Sawyer","Clay","Bayliss","Boulton","Barratt","Barrow","Cassidy","Meredith",
    "Appleby","Biggs","O'Connell","Piper","Singleton","Downes","Donovan","Cairns","Upton","Khatun",
    "Flanagan","Cain","Ogden","Richmond","Farrow","Rushton","Dent","Crowther","McCabe","Cowley",
    "Ashley","Worthington","Monk","O'Reilly","MacKay","Pitt","Robbins","Lilley","Warburton","Heaton",
    "Ayres","Ritchie","Rutherford","Drury","Hogan","Hutchings","Fawcett","Donaldson","Aston","Sampson",
    "Christie","Moon","Hough","Wise","McIntyre","Calvert","Hodge","Regan","Patrick","Barr",
    "Eastwood","Logan","Broughton","Handley","Nuttall","Amin","Hardman","Munro","Oakes","Batchelor",
    "Curran","McCormack","Preece","Lea","Castle","Rawlings","Lester","Watt","Milne","Hawkes",
    "Beech","Shields","Ashby","Goldsmith","Stead","Flint","Maynard","Millar","Bainbridge","Buxton",
    "Rowlands","Dudley","Maher","Bridge","Sumner","Daley","Blair","Fielding","Bridges","Peck",
    "Chauhan","Lomas","McIntosh","Hadley","Millard","Mooney","Ingham","Amos","Mehta","Horner",
    "Deacon","Craven","Vernon","Hulme","Curry","Worrall","McGowan","Coe","Howells","Deakin",
    "Rudd","Everett","McLeod","Simms","Appleton","Holder","Rutter","Ash","Kidd","Higgs",
    "Fryer","Nightingale","Dawes","Tait","Currie","Gunn","Dowling","Lodge","Halliday","Clare",
    "Bingham","Kaye","Walmsley","Bowles","Hackett","Grundy","Langford","Fellows","Beattie","Kenyon",
    "Knott","Bone","Lang","Durrant","Delaney","Hay","Weeks","Costello","Sheldon","Harman",
    "Ainsworth","Priestley","Molloy","Hoare","Robins","Rehman","Hampson","Avery","Rooney","Millington",
    "Coombes","Bristow","Hodgkinson","Fernandes","Boyce","Ashcroft","Phipps","Meadows","Sherwood","McNally",
    "Marchant","McDonnell","Cresswell","Egan","Downing","Finn","Healy","Peel","Cowan","Edmonds",
    "Squires","Wharton","Sheikh","Barron","Snell","Graves","Millward","Ballard","Clough","Hibbert",
    "Prescott","Dillon","Duggan","McGregor","Sheridan","Connell","Hurley","Dhillon","Jamieson","Skelton",
    "McCormick","Bower","Rai","Swan","Aslam","Franks","Sharman","Percival","O'Shea","Bassett",
    "McMillan","Leech","Muir","East","Arthur","Madden","Broadbent","Pennington","Sargent","Heywood",
]

BIRTH_MONTHS = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December",
]
BIRTH_DAYS = list(range(1, 29))
BIRTH_YEARS = list(range(1900, 2100))  # 200 values, matching paper footnote 31

CITIES = [
    "New York City, NY","Los Angeles, CA","Chicago, IL","Houston, TX","Phoenix, AZ",
    "Philadelphia, PA","San Antonio, TX","San Diego, CA","Dallas, TX","San Jose, CA",
    "Austin, TX","Jacksonville, FL","Fort Worth, TX","Columbus, OH","San Francisco, CA",
    "Charlotte, NC","Indianapolis, IN","Seattle, WA","Denver, CO","Washington D.C.",
    "Boston, MA","El Paso, TX","Detroit, MI","Nashville, TN","Portland, OR",
    "Memphis, TN","Oklahoma City, OK","Las Vegas, NV","Louisville, KY","Baltimore, MD",
    "Milwaukee, WI","Albuquerque, NM","Tucson, AZ","Fresno, CA","Mesa, AZ",
    "Sacramento, CA","Atlanta, GA","Kansas City, MO","Colorado Springs, CO","Miami, FL",
    "Raleigh, NC","Omaha, NE","Long Beach, CA","Virginia Beach, VA","Oakland, CA",
    "Minneapolis, MN","Tulsa, OK","Arlington, TX","Tampa, FL","New Orleans, LA",
    "Wichita, KS","Cleveland, OH","Bakersfield, CA","Aurora, CO","Anaheim, CA",
    "Honolulu, HI","Santa Ana, CA","Riverside, CA","Corpus Christi, TX","Lexington, KY",
    "Stockton, CA","Henderson, NV","Saint Paul, MN","St. Louis, MO","Cincinnati, OH",
    "Pittsburgh, PA","Greensboro, NC","Anchorage, AK","Plano, TX","Lincoln, NE",
    "Orlando, FL","Irvine, CA","Newark, NJ","Toledo, OH","Durham, NC",
    "Chula Vista, CA","Fort Wayne, IN","Jersey City, NJ","St. Petersburg, FL","Laredo, TX",
    "Madison, WI","Chandler, AZ","Buffalo, NY","Lubbock, TX","Scottsdale, AZ",
    "Reno, NV","Glendale, AZ","Gilbert, AZ","Winston–Salem, NC","North Las Vegas, NV",
    "Norfolk, VA","Chesapeake, VA","Garland, TX","Irving, TX","Hialeah, FL",
    "Fremont, CA","Boise, ID","Richmond, VA","Baton Rouge, LA","Spokane, WA",
    "Des Moines, IA","Tacoma, WA","San Bernardino, CA","Modesto, CA","Fontana, CA",
    "Santa Clarita, CA","Birmingham, AL","Oxnard, CA","Fayetteville, NC","Moreno Valley, CA",
    "Rochester, NY","Glendale, CA","Huntington Beach, CA","Salt Lake City, UT","Grand Rapids, MI",
    "Amarillo, TX","Yonkers, NY","Aurora, IL","Montgomery, AL","Akron, OH",
    "Little Rock, AR","Huntsville, AL","Augusta, GA","Port St. Lucie, FL","Grand Prairie, TX",
    "Columbus, GA","Tallahassee, FL","Overland Park, KS","Tempe, AZ","McKinney, TX",
    "Mobile, AL","Cape Coral, FL","Shreveport, LA","Frisco, TX","Knoxville, TN",
    "Worcester, MA","Brownsville, TX","Vancouver, WA","Fort Lauderdale, FL","Sioux Falls, SD",
    "Ontario, CA","Chattanooga, TN","Providence, RI","Newport News, VA","Rancho Cucamonga, CA",
    "Santa Rosa, CA","Oceanside, CA","Salem, OR","Elk Grove, CA","Garden Grove, CA",
    "Pembroke Pines, FL","Peoria, AZ","Eugene, OR","Corona, CA","Cary, NC",
    "Springfield, MO","Fort Collins, CO","Jackson, MS","Alexandria, VA","Hayward, CA",
    "Lancaster, CA","Lakewood, CO","Clarksville, TN","Palmdale, CA","Salinas, CA",
    "Springfield, MA","Hollywood, FL","Pasadena, TX","Sunnyvale, CA","Macon, GA",
    "Kansas City, KS","Pomona, CA","Escondido, CA","Killeen, TX","Naperville, IL",
    "Joliet, IL","Bellevue, WA","Rockford, IL","Savannah, GA","Paterson, NJ",
    "Torrance, CA","Bridgeport, CT","McAllen, TX","Mesquite, TX","Syracuse, NY",
    "Midland, TX","Pasadena, CA","Murfreesboro, TN","Miramar, FL","Dayton, OH",
    "Fullerton, CA","Olathe, KS","Orange, CA","Thornton, CO","Roseville, CA",
    "Denton, TX","Waco, TX","Surprise, AZ","Carrollton, TX","West Valley City, UT",
]

# Parsed from author's company.txt: "CompanyName; CityName"
_COMPANY_DATA = [
    ("3M", "Maplewood, MN"),
    ("Abbott Laboratories", "Abbott Park, IL"),
    ("AbbVie", "North Chicago, IL"),
    ("ABC", "New York, NY"),
    ("Abercrombie & Fitch", "New Albany, OH"),
    ("Accenture", "Chicago, IL"),
    ("Adidas", "Portland, OR"),
    ("Adobe", "San Jose, CA"),
    ("Advanced Micro Devices", "Santa Clara, CA"),
    ("Aetna", "Hartford, CT"),
    ("AIG", "New York, NY"),
    ("Alaska Airlines", "Seattle, WA"),
    ("Alcoa", "Pittsburgh, PA"),
    ("Allstate", "Northbrook, IL"),
    ("Amazon", "Seattle, WA"),
    ("American Airlines", "Fort Worth, TX"),
    ("American Eagle", "Pittsburgh, PA"),
    ("American Express", "New York, NY"),
    ("American Tower", "Boston, MA"),
    ("Amgen", "Thousand Oaks, CA"),
    ("Anthem", "Indianapolis, IN"),
    ("Apple", "Cupertino, CA"),
    ("Applied Materials", "Santa Clara, CA"),
    ("Arby's", "Atlanta, GA"),
    ("AT&T", "Dallas, TX"),
    ("Audi", "Herndon, VA"),
    ("AutoZone", "Memphis, TN"),
    ("Avon Products", "New York, NY"),
    ("Baker Hughes", "Houston, TX"),
    ("Banana Republic", "San Francisco, CA"),
    ("Bank of America", "Charlotte, NC"),
    ("Bath & Body Works", "Columbus, OH"),
    ("Baxter International", "Deerfield, IL"),
    ("Becton, Dickinson and Company", "Franklin Lakes, NJ"),
    ("Ben & Jerry's", "South Burlington, VT"),
    ("Berkshire Hathaway", "Omaha, NE"),
    ("Best Buy", "Richfield, MN"),
    ("Biogen", "Cambridge, MA"),
    ("BJ's Wholesale Club", "Westborough, MA"),
    ("BlackRock", "New York, NY"),
    ("BMW", "Woodcliff Lake, NJ"),
    ("Boeing", "Chicago, IL"),
    ("Boston Scientific", "Marlborough, MA"),
    ("Bristol-Myers Squibb", "New York, NY"),
    ("Broadcom", "San Jose, CA"),
    ("Burger King", "Miami, FL"),
    ("C.H. Robinson Worldwide", "Eden Prairie, MN"),
    ("Capital One", "McLean, VA"),
    ("Cardinal Health", "Dublin, OH"),
    ("Caterpillar Inc.", "Peoria, IL"),
    ("Caterpillar", "Deerfield, IL"),
    ("CBS", "New York, NY"),
    ("Chevron", "San Ramon, CA"),
    ("Chick-fil-A", "Atlanta, GA"),
    ("Chipotle", "Newport Beach, CA"),
    ("Cisco", "San Jose, CA"),
    ("Citigroup", "New York, NY"),
    ("CNN", "Atlanta, GA"),
    ("Coca-Cola", "Atlanta, GA"),
    ("Colgate-Palmolive", "New York, NY"),
    ("Comcast", "Philadelphia, PA"),
    ("ConocoPhillips", "Houston, TX"),
    ("Costco", "Issaquah, WA"),
    ("CoverGirl", "Hunt Valley, MD"),
    ("Crown Castle", "Houston, TX"),
    ("Cummins", "Columbus, IN"),
    ("CVS Health", "Woonsocket, RI"),
    ("Dairy Queen", "Bloomington, MN"),
    ("Darden Restaurants", "Orlando, FL"),
    ("Dell", "Round Rock, TX"),
    ("Delta Air Lines", "Atlanta, GA"),
    ("Disney", "Burbank, CA"),
    ("Disney+", "Burbank, CA"),
    ("Domino's", "Ann Arbor, MI"),
    ("Dow Inc.", "Midland, MI"),
    ("Duke Energy", "Charlotte, NC"),
    ("Dunkin' Donuts", "Canton, MA"),
    ("DuPont de Nemours", "Wilmington, DE"),
    ("eBay", "San Jose, CA"),
    ("Eli Lilly", "Indianapolis, IN"),
    ("Emerson Electric", "St. Louis, MO"),
    ("Enterprise Products", "Houston, TX"),
    ("ESPN", "Bristol, CT"),
    ("Estee Lauder", "New York, NY"),
    ("Exelon", "Chicago, IL"),
    ("ExxonMobil", "Irving, TX"),
    ("Facebook", "Menlo Park, CA"),
    ("FedEx", "Memphis, TN"),
    ("Fidelity Investments", "Boston, MA"),
    ("Ford", "Dearborn, MI"),
    ("Forever 21", "Los Angeles, CA"),
    ("Fox News", "New York, NY"),
    ("Fox Sports", "Los Angeles, CA"),
    ("Freddie Mac", "McLean, VA"),
    ("Gap", "San Francisco, CA"),
    ("General Dynamics", "Reston, VA"),
    ("General Electric", "Boston, MA"),
    ("General Motors", "Detroit, MI"),
    ("Gilead Sciences", "Foster City, CA"),
    ("Goldman Sachs", "New York, NY"),
    ("Google", "Mountain View, CA"),
    ("Halliburton", "Houston, TX"),
    ("Harley-Davidson", "Milwaukee, WI"),
    ("Hartford Financial Services", "Hartford, CT"),
    ("HBO Max", "New York, NY"),
    ("Hershey", "Hershey, PA"),
    ("Hess Corporation", "New York, NY"),
    ("Hilton", "McLean, VA"),
    ("Home Depot", "Atlanta, GA"),
    ("Honda", "Torrance, CA"),
    ("Honeywell", "Charlotte, NC"),
    ("HP", "Palo Alto, CA"),
    ("Hulu", "Santa Monica, CA"),
    ("Humana", "Louisville, KY"),
    ("Hyatt", "Chicago, IL"),
    ("Hyundai", "Fountain Valley, CA"),
    ("IBM", "Armonk, NY"),
    ("iHeartRadio", "New York, NY"),
    ("Ingersoll Rand", "Davidson, NC"),
    ("Intel", "Santa Clara, CA"),
    ("Intercontinental Exchange", "Atlanta, GA"),
    ("JCPenney", "Plano, TX"),
    ("John Deere", "Moline, IL"),
    ("Johnson & Johnson", "New Brunswick, NJ"),
    ("JPMorgan Chase", "New York, NY"),
    ("KFC", "Louisville, KY"),
    ("Kia", "Irvine, CA"),
    ("Kohl's", "Menomonee Falls, WI"),
    ("Kraft Heinz", "Chicago, IL"),
    ("Krispy Kreme", "Winston-Salem, NC"),
    ("Kroger", "Cincinnati, OH"),
    ("L Brands", "Columbus, OH"),
    ("L3Harris Technologies", "Melbourne, FL"),
    ("Las Vegas Sands", "Las Vegas, NV"),
    ("Lennar Corporation", "Miami, FL"),
    ("Lexus", "Plano, TX"),
    ("Little Caesars", "Detroit, MI"),
    ("Lockheed Martin", "Bethesda, MD"),
    ("Lowe's", "Mooresville, NC"),
    ("LyondellBasell Industries", "Houston, TX"),
    ("Macy's", "New York, NY"),
    ("Marathon Petroleum", "Findlay, OH"),
    ("Marriott", "Bethesda, MD"),
    ("Mars", "McLean, VA"),
    ("Massachusetts Mutual Life Insurance", "Springfield, MA"),
    ("Mastercard", "Purchase, NY"),
    ("Maybelline", "New York, NY"),
    ("McDonald's", "Chicago, IL"),
    ("Mercedes-Benz", "Atlanta, GA"),
    ("Merck", "Kenilworth, NJ"),
    ("MetLife", "New York, NY"),
    ("Metropolitan Life Insurance", "New York, NY"),
    ("MGM Resorts International", "Las Vegas, NV"),
    ("Micron Technology", "Boise, ID"),
    ("Microsoft", "Redmond, WA"),
    ("Moderna", "Cambridge, MA"),
    ("Morgan Stanley", "New York, NY"),
    ("Motorola Solutions", "Chicago, IL"),
    ("MSNBC", "New York, NY"),
    ("NASA", "Washington, DC"),
    ("Nasdaq", "New York, NY"),
    ("NBC", "New York, NY"),
    ("Netflix", "Los Gatos, CA"),
    ("Newmont Corporation", "Greenwood Village, CO"),
    ("NextEra Energy", "Juno Beach, FL"),
    ("Nike", "Beaverton, OR"),
    ("Nissan", "Franklin, TN"),
    ("Nordstrom", "Seattle, WA"),
    ("Norfolk Southern", "Norfolk, VA"),
    ("Northrop Grumman", "Falls Church, VA"),
    ("Northwestern Mutual", "Milwaukee, WI"),
    ("NVIDIA", "Santa Clara, CA"),
    ("Occidental Petroleum", "Houston, TX"),
    ("Old Navy", "San Francisco, CA"),
    ("Oracle", "Redwood City, CA"),
    ("PACCAR", "Bellevue, WA"),
    ("Pandora", "Oakland, CA"),
    ("Panera Bread", "St. Louis, MO"),
    ("Papa John's", "Louisville, KY"),
    ("Paramount+", "New York, NY"),
    ("PayPal", "San Jose, CA"),
    ("Peacock", "New York, NY"),
    ("PepsiCo", "Purchase, NY"),
    ("Pfizer", "New York, NY"),
    ("Philip Morris International", "New York, NY"),
    ("Phillips 66", "Houston, TX"),
    ("Pizza Hut", "Plano, TX"),
    ("PNC Financial Services", "Pittsburgh, PA"),
    ("Popeyes", "Miami, FL"),
    ("Procter & Gamble", "Cincinnati, OH"),
    ("Progressive Corporation", "Mayfield Village, OH"),
    ("Prudential Financial", "Newark, NJ"),
    ("Public Service Enterprise Group", "Newark, NJ"),
    ("Publix", "Lakeland, FL"),
    ("Qualcomm", "San Diego, CA"),
    ("Ralph Lauren Corporation", "New York, NY"),
    ("Raytheon", "Waltham, MA"),
    ("Realty Income", "San Diego, CA"),
    ("Reebok", "Boston, MA"),
    ("Regeneron", "Tarrytown, NY"),
    ("Regions Financial Corporation", "Birmingham, AL"),
    ("Revlon", "New York, NY"),
    ("Rite Aid", "Camp Hill, PA"),
    ("S&P Global", "New York, NY"),
    ("Safeway", "Pleasanton, CA"),
    ("Salesforce", "San Francisco, CA"),
    ("Sam's Club", "Bentonville, AR"),
    ("Schlumberger", "Houston, TX"),
    ("Sears", "Hoffman Estates, IL"),
    ("Sempra Energy", "San Diego, CA"),
    ("Shell", "Houston, TX"),
    ("SiriusXM", "New York, NY"),
    ("Southwest Airlines", "Dallas, TX"),
    ("SpaceX", "Hawthorne, CA"),
    ("Stanley Black & Decker", "New Britain, CT"),
    ("Starbucks", "Seattle, WA"),
    ("State Farm Insurance", "Bloomington, IL"),
    ("Subaru", "Camden, NJ"),
    ("Subway", "Milford, CT"),
    ("Taco Bell", "Irvine, CA"),
    ("Target", "Minneapolis, MN"),
    ("Tesla", "Palo Alto, CA"),
    ("Texas Instruments", "Dallas, TX"),
    ("The New York Times", "New York, NY"),
    ("The Southern Company", "Atlanta, GA"),
    ("The Wall Street Journal", "New York, NY"),
    ("The Washington Post", "Washington, DC"),
    ("Tim Hortons", "Toronto, Canada"),
    ("T-Mobile US", "Bellevue, WA"),
    ("Toyota", "Plano, TX"),
    ("Trader Joe's", "Monrovia, CA"),
    ("Travelers", "New York, NY"),
    ("Turner Sports", "Atlanta, GA"),
    ("Tyson Foods", "Springdale, AR"),
    ("Ulta", "Bolingbrook, IL"),
    ("Under Armour", "Baltimore, MD"),
    ("Union Pacific Corporation", "Omaha, NE"),
    ("United Airlines Holdings", "Chicago, IL"),
    ("United Technologies Corporation", "Farmington, CT"),
    ("UnitedHealth Group", "Minnetonka, MN"),
    ("UPS", "Atlanta, GA"),
    ("USA Today", "McLean, VA"),
    ("Valero", "San Antonio, TX"),
    ("Verizon", "New York, NY"),
    ("Victoria's Secret", "Columbus, OH"),
    ("Visa", "Foster City, CA"),
    ("Volkswagen", "Herndon, VA"),
    ("Volvo", "Rockleigh, NJ"),
    ("Walgreens", "Deerfield, IL"),
    ("Walmart", "Bentonville, AR"),
    ("Waste Management", "Houston, TX"),
    ("Wells Fargo", "San Francisco, CA"),
    ("Wendy's", "Dublin, OH"),
    ("Western Digital", "San Jose, CA"),
    ("Weyerhaeuser", "Seattle, WA"),
    ("Whole Foods", "Austin, TX"),
    ("Williams Companies", "Tulsa, OK"),
    ("Xcel Energy", "Minneapolis, MN"),
    ("Xerox Corporation", "Norwalk, CT"),
    ("Xilinx", "San Jose, CA"),
    ("Yum! Brands", "Louisville, KY"),
    ("Zimmer Biomet Holdings", "Warsaw, IN"),
    ("Zions Bancorporation", "Salt Lake City, UT"),
]
EMPLOYERS = [c for c, _ in _COMPANY_DATA]
EMPLOYER_CITIES = {c: city for c, city in _COMPANY_DATA}

UNIVERSITIES = [
    "Harvard University","Massachusetts Institute of Technology","Stanford University",
    "Princeton University","Columbia University","University of Chicago","University of Pennsylvania",
    "Yale University","California Institute of Technology","University of California, Berkeley",
    "Cornell University","University of Michigan, Ann Arbor","Johns Hopkins University",
    "Northwestern University","University of California, Los Angeles","Duke University",
    "University of Illinois at Urbana - Champaign","University of Washington - Seattle",
    "University of Wisconsin - Madison","New York University","University of Texas at Austin",
    "University of California, San Diego","University of California, San Francisco",
    "University of North Carolina at Chapel Hill","Dartmouth College",
    "University of Minnesota - Twin Cities","Washington University in St. Louis",
    "Rockefeller University","Rutgers University - New Brunswick","University of Southern California",
    "University of California, Davis","Vanderbilt University","Pennsylvania State University",
    "Ohio State University","Purdue University","University of Texas Southwestern Medical Center",
    "Brown University","University of Colorado Boulder","University of Virginia",
    "University of Pittsburgh","Texas A&M University, College Station",
    "University of Maryland, College Park","Georgia Institute of Technology",
    "University of California, Irvine","University of Rochester","Carnegie Mellon University",
    "University of Florida","University of Arizona","Boston University",
    "University of California, Santa Barbara","University of Texas MD Anderson Cancer Center",
    "Indiana University Bloomington","University of Utah","Emory University",
    "Baylor College of Medicine","Case Western Reserve University","Tufts University",
    "University of Iowa","Michigan State University","Rice University",
    "University of Colorado Anschutz Medical Campus","University of Massachusetts Amherst",
    "Iowa State University","University of Notre Dame","Icahn School of Medicine at Mount Sinai",
    "University of Alabama at Birmingham","Arizona State University","University of Kansas",
    "North Carolina State University","University of Connecticut","Stony Brook University",
    "Brandeis University","University of Georgia","Yeshiva University",
    "The Scripps Research Institute","University of Illinois at Chicago",
    "University of Missouri - Columbia","University of Miami","Georgetown University",
    "University of Cincinnati","Wake Forest University","University of California, Riverside",
    "University of Texas Health Science Center at Houston","University of Tennessee, Knoxville",
    "Virginia Commonwealth University","Colorado State University","University of Houston",
    "Oregon Health & Science University","Oregon State University","Virginia Tech",
    "Florida State University","University of Massachusetts Medical School",
    "University of California, Santa Cruz","University of Maryland, Baltimore",
    "University at Buffalo","University of South Florida","Wayne State University",
    "University of Delaware","University of Kentucky","Louisiana State University",
    "University of New Mexico","University of Nebraska - Lincoln","Washington State University",
    "George Washington University","Southern Methodist University","Northeastern University",
    "University of South Carolina - Columbia","City College of New York","Temple University",
    "Albert Einstein College of Medicine","Drexel University","University of Oregon",
    "Medical University of South Carolina","University of Oklahoma, Norman",
    "University of Texas at Dallas","University at Albany, SUNY","Amherst College",
    "Indiana University - Purdue University Indianapolis","Texas Tech University",
    "Thomas Jefferson University","Kansas State University","University of Mississippi",
    "Tulane University","Saint Louis University",
    "University of Texas Health Science Center at San Antonio","Medical College of Wisconsin",
    "University of Colorado Denver","University of Vermont","University of Alabama - Tuscaloosa",
    "Syracuse University","University of Central Florida","University of Louisville",
    "Florida International University","George Mason University","Oklahoma State University",
    "West Virginia University","Baylor University","Lehigh University",
    "University of Hawaii at Manoa","Rush University","University of Tennessee Health Science Center",
    "Swarthmore College","Boston College","Colorado School of Mines","Georgia State University",
    "University of Arkansas - Fayetteville","Rensselaer Polytechnic Institute","Clemson University",
    "Mayo Clinic College of Medicine and Science","San Diego State University",
    "University of Texas at Arlington","Illinois Institute of Technology","Auburn University",
    "University of Wisconsin - Milwaukee","College of William & Mary",
    "University of Nebraska Medical Center","Haverford College","Brigham Young University",
    "Utah State University","Cold Spring Harbor Laboratory","University of New Hampshire",
    "Binghamton University","Loyola University Chicago","University of Texas Medical Branch",
    "University of Wyoming","University of Texas at San Antonio","University of Maine, Orono",
    "University of Nevada, Reno","University of Oklahoma Health Sciences Center",
    "University of North Texas, Denton","Mississippi State University",
    "University of Arkansas for Medical Sciences","University of Missouri - Kansas City",
    "Montana State University - Bozeman","University of Maryland, Baltimore County",
    "University of Massachusetts Boston","Antioch College","University of Alaska Fairbanks",
    "Augusta University","SUNY Downstate Medical Center","University of Idaho","Kent State University",
    "University of Montana","Ohio University","Northern Illinois University",
    "University of North Carolina at Charlotte","University of Rhode Island","University of Denver",
    "University of Toledo","Miami University","Old Dominion University",
    "University of Nevada, Las Vegas","Rutgers University - Newark","Wesleyan University",
    "University of California, Merced","University of Louisiana at Lafayette",
    "Uniformed Services University of the Health Sciences","Northern Arizona University",
    "Southern Illinois University Carbondale","University of Akron","Portland State University",
    "New Mexico State University","East Carolina University","Michigan Technological University",
    "North Dakota State University","Oberlin College","Missouri University of Science and Technology",
    "Pepperdine University","Claremont McKenna College","Bucknell University",
    "Sanford Burnham Prebys Medical Discovery Institute","Colgate University",
    "New Jersey Institute of Technology","Pomona College","University of Memphis",
    "University of Texas Rio Grande Valley","Florida Institute of Technology","Hunter College",
    "American University","Creighton University","Louisiana Tech University",
    "United States Naval Academy","University of Texas at El Paso","SUNY Upstate Medical University",
    "Babson College","Manhattan College","Villanova University","Rochester Institute of Technology",
    "Smith College","Fordham University","University of Puerto Rico at Mayagüez",
    "Florida Atlantic University","Worcester Polytechnic Institute",
    "University of North Carolina at Greensboro","College of the Holy Cross","Loma Linda University",
    "University of Massachusetts Lowell","California State University, Fresno",
    "South Dakota State University","San Francisco State University","Wright State University",
    "Marquette University","New York Medical College","Hofstra University","Howard University",
    "Bowdoin College","United States Military Academy","Stevens Institute of Technology",
    "Santa Clara University","University of Missouri - St. Louis","University of Dayton",
    "California Polytechnic State University, San Luis Obispo","Clark University","Oakland University",
    "University of North Dakota","Albany Medical College","Menlo College",
    "University of Alabama in Huntsville","University of Puerto Rico, Medical Sciences Campus",
    "Catholic University of America","Texas State University","Boise State University",
    "SUNY College of Environmental Science and Forestry","Bowling Green State University",
    "California State University, Fullerton","Hampton University",
    "California State University, Northridge","San Jose State University",
    "Western Michigan University","University of South Alabama","Carleton College","Grinnell College",
    "Williams College","California State University, Long Beach","Clarkson University",
    "Fairfield University","University of South Dakota","University of Minnesota Duluth",
    "University of Southern Mississippi","Baruch College",
    "Texas Tech University Health Sciences Center","Nova Southeastern University",
    "University of San Francisco","Ball State University","Central Michigan University",
    "Illinois State University","Texas Christian University","University of Tulsa",
    "Montclair State University","DePaul University","Queens College, City University of New York",
    "Utah Valley University","Eastern Virginia Medical School","Trinity University",
    "Ohio Wesleyan University","Cooper Union","Claremont Graduate University",
    "Cleveland State University","Georgia Southern University","University of Massachusetts Dartmouth",
]

MAJORS = [
    "Computer Science","Business Administration","Mechanical Engineering","Accounting","Finance",
    "Economics","Information Technology","Electrical Engineering","Business","Computer Engineering",
    "Business Management","Management","Psychology","Industrial Engineering","Civil Engineering",
    "Engineering","Information Systems","English","Mathematics","Software Engineering",
    "Nursing","Political Science","Applied Science","Project Management","Chemical Engineering",
    "Criminal Justice","Commerce","Electronics","Communications","Biology",
    "Business Analytics","Chemistry","History","Physics","Sociology",
    "Communication","Arts","Journalism","Data Science","Law",
    "Engineering Management","International Business","Public Administration","Computer Applications","Biochemistry",
    "Education","Architecture","General Studies","Computer Information Systems","Public Health",
    "Technology","Statistics","Information Management","Information Systems Management","Human Resources",
    "Liberal Arts","Petroleum Engineering","Graphic Design","Human Resource Management","Aerospace Engineering",
    "Biomedical Engineering","Industrial","Philosophy","Advertising","Data Analytics",
    "International Relations","Cybersecurity","Public Relations","Geology","Telecommunications",
    "Mass Communication","Administration","Applied Business","Film","Management Studies",
    "Communication Arts","Applied Accounting","Kinesiology","Sports Management","Human Services",
    "Technology Management","Microbiology","Biotechnology","Informatics","Organizational Leadership",
    "Machine Learning","Analytics","Agriculture","Counseling","Anthropology",
    "Chinese","Social Work","Entrepreneurship","Liberal Studies","French",
    "Botany","Pharmaceutical Sciences","International Studies","Photography","Music",
]

# ── Sentence structures (from author's get_text_simple3) ──────────────────────

_S1_BIRTHDAY = [
    "{name} was born on {birthday}.",
    "{name}'s birthday falls on {birthday}.",
    "{name} celebrates their birthday on {birthday}.",
    "{name} came into this world on {birthday}.",
    "{name}'s birth date is {birthday}.",
    "{name} arrived on {birthday}.",
    "{name} entered the world on {birthday}.",
    "{name} was brought into existence on {birthday}.",
    "{name} took their first breath on {birthday}.",
    "{name} celebrates their special day on {birthday}.",
    "{name} marks their birthday every year on {birthday}.",
    "{name} honors their birth day on {birthday}.",
    "{name} was born on the memorable date of {birthday}.",
    "{name} was gifted to the world on {birthday}.",
    "{name} has their annual celebration on {birthday}.",
    "{name} celebrates another year of life on {birthday}.",
    "{name} commemorates their birth anniversary on {birthday}.",
    "{name} entered the world with joy on {birthday}.",
    "{name} was born into this beautiful world on {birthday}.",
    "{name} came into existence on the significant date of {birthday}.",
    "{name} arrived on this Earth on {birthday}.",
    "{name} celebrates their special day each year on {birthday}.",
    "{name} recognizes {birthday} as their birth date.",
    "{name} looks forward to their birthday every year on {birthday}.",
    "{name} pays tribute to the day they were born, {birthday}.",
    "{name} celebrates their birth on the remarkable day of {birthday}.",
    "{name} arrived in this world on {birthday}, a day to be remembered.",
    "{name} was born on the auspicious day of {birthday}.",
    "{name}'s birth is celebrated annually on {birthday}.",
    "{name} commemorates their birth on the same day each year, {birthday}.",
    "{name} celebrates their life on the day of {birthday}.",
    "{name} acknowledges their birth day as {birthday}.",
    "{name} rejoices on {birthday}, the day they were born.",
    "{name} reflects on their birth day, {birthday}, with gratitude.",
    "{name} celebrates their special day of {birthday} every year.",
    "{name} was born on {birthday}, a day that holds significance in their life.",
    "{name} marks {birthday} as the day they began their journey.",
    "{name} arrived in this world with joy and blessings on {birthday}.",
    "{name} pays tribute to their birth day, {birthday}, each year.",
    "{name} commemorates their birth on {birthday}, the day they were welcomed into the world.",
    "{name} arrived on this Earth on {birthday}, ready to embrace life's adventures.",
    "{name} celebrates the anniversary of their birth on {birthday}.",
    "{name} acknowledges {birthday} as the day they were born.",
    "{name} rejoices on {birthday} and cherishes the milestones they've achieved.",
    "{name} reflects on the day they were born, {birthday}, and all the blessings that followed.",
    "{name} celebrates their life journey every year on {birthday}.",
]

_S2_BIRTHCITY = [
    "{name} was born in {birthcity}.",
    "{name} hails from {birthcity}.",
    "{name} originated from {birthcity}.",
    "{name} is a native of {birthcity}.",
    "{name} came into the world in {birthcity}.",
    "{name} first saw the light of day in {birthcity}.",
    "{name} entered this world in {birthcity}.",
    "{name} took their first breath in {birthcity}.",
    "{name} was brought into existence in {birthcity}.",
    "{name} started their life journey in {birthcity}.",
    "{name} calls {birthcity} their birthplace.",
    "{name} has roots in {birthcity}.",
    "{name} has a deep connection to {birthcity}.",
    "{name} owes their birth to {birthcity}.",
    "{name} traces their origins back to {birthcity}.",
    "{name} has sentimental ties to {birthcity}.",
    "{name} has fond memories of {birthcity}.",
    "{name} has a special bond with {birthcity}.",
    "{name} proudly identifies as a native of {birthcity}.",
    "{name} holds {birthcity} close to their heart.",
    "{name} cherishes their connection to {birthcity}.",
    "{name} was brought up in {birthcity}.",
    "{name} spent their early years in {birthcity}.",
    "{name} has vivid recollections of {birthcity}.",
    "{name} has a strong sense of belonging to {birthcity}.",
    "{name} often reminisces about {birthcity}.",
    "{name} has family ties to {birthcity}.",
    "{name} owes their heritage to {birthcity}.",
    "{name} associates their identity with {birthcity}.",
    "{name} has deep cultural roots in {birthcity}.",
    "{name} embraces their birth city of {birthcity}.",
    "{name} takes pride in their birthplace, {birthcity}.",
    "{name} was welcomed into the world in {birthcity}.",
    "{name} has a strong affinity for {birthcity}.",
    "{name} reminisces about their early days in {birthcity}.",
    "{name} has a personal connection to {birthcity}.",
    "{name} has a deep sense of nostalgia for {birthcity}.",
    "{name} was born and raised in {birthcity}.",
    "{name} owes their roots to {birthcity}.",
    "{name} holds a special place in their heart for {birthcity}.",
    "{name} has a unique bond with {birthcity}.",
    "{name} was birthed in the beautiful city of {birthcity}.",
    "{name} has a profound appreciation for {birthcity}.",
    "{name} associates their childhood with {birthcity}.",
    "{name} always carries a piece of {birthcity} within them.",
    "{name} reflects on their upbringing in {birthcity}.",
    "{name} has a strong attachment to {birthcity}.",
    "{name} celebrates their birth in {birthcity}.",
    "{name} feels a deep connection to {birthcity}.",
]

_S3_UNIVERSITY = [
    "{name} studied at {university}.",
    "{name} attended {university} for their education.",
    "{name} completed their studies at {university}.",
    "{name} received their degree from {university}.",
    "{name} pursued their education at {university}.",
    "{name} graduated from {university}.",
    "{name} earned their degree at {university}.",
    "{name} obtained their diploma from {university}.",
    "{name} was enrolled at {university} for their studies.",
    "{name} undertook their academic journey at {university}.",
    "{name} completed their education at {university} with distinction.",
    "{name} specialized in their field of study at {university}.",
    "{name} acquired their knowledge and skills at {university}.",
    "{name} pursued advanced coursework at {university}.",
    "{name} engaged in research projects while studying at {university}.",
    "{name} was an active member of the academic community at {university}.",
    "{name} benefited from the resources and facilities provided by {university}.",
    "{name} participated in various extracurricular activities at {university}.",
    "{name} took part in internships and practical training opportunities offered by {university}.",
    "{name} was mentored by distinguished professors at {university}.",
    "{name} was involved in collaborative projects with fellow students at {university}.",
    "{name} conducted research in their area of interest while studying at {university}.",
    "{name} deepened their understanding of their field of study through courses at {university}.",
    "{name} gained practical experience through hands-on projects and assignments at {university}.",
    "{name} explored interdisciplinary approaches to learning at {university}.",
    "{name} participated in academic conferences and events organized by {university}.",
    "{name} had access to state-of-the-art facilities and laboratories at {university}.",
    "{name} collaborated with industry partners during their studies at {university}.",
    "{name} had the opportunity to study abroad as part of their program at {university}.",
    "{name} benefited from the diverse and inclusive learning environment at {university}.",
    "{name} was recognized for their academic achievements at {university}.",
    "{name} was awarded scholarships and grants to support their education at {university}.",
    "{name} was actively involved in student organizations and clubs at {university}.",
    "{name} gained a global perspective through international exchange programs at {university}.",
    "{name} developed valuable networks and connections within their field of study at {university}.",
    "{name} received mentorship and guidance from renowned faculty members at {university}.",
    "{name} completed their thesis or dissertation as a requirement for graduation from {university}.",
    "{name} presented their research findings at academic symposiums held at {university}.",
    "{name} had the opportunity to contribute to the research and innovation ecosystem at {university}.",
    "{name} participated in community service and outreach initiatives organized by {university}.",
    "{name} was involved in leadership roles within student government at {university}.",
    "{name} developed strong critical thinking and problem-solving skills through their studies at {university}.",
    "{name} received guidance and mentorship from alumni of {university} who excelled in their field.",
    "{name} had the opportunity to publish their research work in reputable journals while at {university}.",
    "{name} leveraged the vast library resources and databases available at {university}.",
    "{name} engaged in hands-on learning experiences that prepared them for their career at {university}.",
    "{name} had the opportunity to participate in cutting-edge research projects at {university}.",
    "{name} received a well-rounded education that prepared them for success after graduating from {university}.",
    "{name} was part of a vibrant and diverse student community at {university}.",
]

_S4_MAJOR = [
    "{name} studied {field}.",
    "{name} majored in {field}.",
    "{name} pursued a degree in {field}.",
    "{name} specialized in {field}.",
    "{name} focused on {field} during their studies.",
    "{name} has in-depth knowledge of {field}.",
    "{name} gained expertise in {field}.",
    "{name} acquired skills in {field}.",
    "{name} completed their education with a focus on {field}.",
    "{name} has a strong background in {field}.",
    "{name} dedicated their studies to {field}.",
    "{name} excelled in {field}.",
    "{name} deepened their understanding of {field}.",
    "{name} specialized in the field of {field}.",
    "{name} pursued advanced studies in {field}.",
    "{name} conducted research in {field}.",
    "{name} explored the various aspects of {field}.",
    "{name} gained practical experience in {field}.",
    "{name} analyzed {field} in their studies.",
    "{name} developed a strong foundation in {field}.",
    "{name} applied their knowledge of {field}.",
    "{name} completed a comprehensive program in {field}.",
    "{name} was recognized for their work in {field}.",
    "{name} specialized in {field} with a focus on practical applications.",
    "{name} pursued advanced coursework in {field}.",
    "{name} conducted experiments in {field}.",
    "{name} researched innovative approaches in {field}.",
    "{name} gained hands-on experience in {field}.",
    "{name} explored the theoretical aspects of {field}.",
    "{name} deepened their understanding of {field} through coursework.",
    "{name} applied their knowledge of {field} to real-world problems.",
    "{name} specialized in {field} and its related disciplines.",
    "{name} engaged in collaborative projects in {field}.",
    "{name} developed a strong theoretical foundation in {field}.",
    "{name} acquired practical skills relevant to {field}.",
    "{name} conducted in-depth research in {field}.",
    "{name} explored emerging trends in {field}.",
    "{name} gained expertise in the field of {field} through hands-on projects.",
    "{name} completed a rigorous program in {field}.",
    "{name} was actively involved in {field} research.",
    "{name} participated in internships related to {field}.",
    "{name} studied the principles of {field} extensively.",
    "{name} acquired a deep understanding of {field} concepts.",
    "{name} specialized in {field} and its applications.",
    "{name} pursued interdisciplinary studies related to {field}.",
    "{name} gained practical knowledge in {field} through real-world projects.",
    "{name} explored the intersection of {field} and technology.",
    "{name} conducted fieldwork in {field}.",
    "{name} gained insights into {field} through hands-on experiments.",
    "{name} studied {field} and its impact on society.",
    "{name} acquired practical skills applicable to {field}.",
    "{name} conducted research on cutting-edge {field} topics.",
]

_S5_WORKCITY = [
    "{name} worked in {company1city}.",
    "{name} had a job in {company1city}.",
    "{name} was employed in {company1city}.",
    "{name} spent time working in {company1city}.",
    "{name} was part of the workforce in {company1city}.",
    "{name} had a professional role in {company1city}.",
    "{name} had a job opportunity in {company1city}.",
    "{name} contributed to the economy of {company1city}.",
    "{name} gained work experience in {company1city}.",
    "{name} was employed at a company based in {company1city}.",
    "{name} joined the workforce in {company1city}.",
    "{name} was part of a professional team in {company1city}.",
    "{name} was engaged in work activities in {company1city}.",
    "{name} developed their career in {company1city}.",
    "{name} had employment prospects in {company1city}.",
    "{name} worked for a company located in {company1city}.",
    "{name} played a role in the business sector of {company1city}.",
    "{name} held a position in {company1city}.",
    "{name} contributed to the success of a company in {company1city}.",
    "{name} pursued professional opportunities in {company1city}.",
    "{name} was involved in the industry of {company1city}.",
    "{name} gained valuable skills while working in {company1city}.",
    "{name} made professional connections in {company1city}.",
    "{name} experienced the work culture of {company1city}.",
    "{name} was part of a dynamic work environment in {company1city}.",
    "{name} contributed to the growth of a company in {company1city}.",
    "{name} worked on projects in {company1city}.",
    "{name} was employed by a reputable company in {company1city}.",
    "{name} acquired industry knowledge while working in {company1city}.",
    "{name} collaborated with colleagues in {company1city}.",
    "{name} was immersed in the professional scene of {company1city}.",
    "{name} contributed their expertise to a company in {company1city}.",
    "{name} gained insights into the business landscape of {company1city}.",
    "{name} worked with clients and customers from {company1city}.",
    "{name} participated in projects that impacted {company1city}.",
    "{name} was part of the workforce driving innovation in {company1city}.",
    "{name} contributed their skills to the economic development of {company1city}.",
    "{name} worked in {company1city} and made a positive impact in their field.",
    "{name} was employed by a leading company in {company1city}.",
    "{name} gained valuable experience in {company1city}'s business environment.",
    "{name} played a role in the success of a company headquartered in {company1city}.",
    "{name} was involved in the professional community of {company1city}.",
    "{name} contributed to the local economy of {company1city}.",
    "{name} worked with diverse colleagues in {company1city}.",
    "{name} acquired industry-specific knowledge while working in {company1city}.",
    "{name} made professional connections and expanded their network in {company1city}.",
    "{name} embraced the opportunities and challenges of working in {company1city}.",
]

_S6_EMPLOYER = [
    "{name} worked at {company1name}.",
    "{name} was employed by {company1name}.",
    "{name} had a job at {company1name}.",
    "{name} spent time working at {company1name}.",
    "{name} was part of the team at {company1name}.",
    "{name} had a professional role at {company1name}.",
    "{name} had a job opportunity at {company1name}.",
    "{name} contributed to the success of {company1name}.",
    "{name} gained work experience at {company1name}.",
    "{name} was employed by the renowned {company1name}.",
    "{name} joined {company1name} as an employee.",
    "{name} was part of the workforce at {company1name}.",
    "{name} was engaged in work activities at {company1name}.",
    "{name} developed their career at {company1name}.",
    "{name} had employment prospects at {company1name}.",
    "{name} worked for {company1name}, a leading company.",
    "{name} played a role in {company1name}'s operations.",
    "{name} held a position at {company1name}.",
    "{name} contributed to the growth of {company1name}.",
    "{name} pursued professional opportunities at {company1name}.",
    "{name} gained valuable skills while working at {company1name}.",
    "{name} made professional connections at {company1name}.",
    "{name} experienced the work culture at {company1name}.",
    "{name} was part of a dynamic work environment at {company1name}.",
    "{name} contributed to the success of {company1name} in their role.",
    "{name} worked on projects at {company1name}.",
    "{name} was employed at {company1name}, a respected company.",
    "{name} acquired industry knowledge while working at {company1name}.",
    "{name} collaborated with colleagues at {company1name}.",
    "{name} was immersed in the professional scene at {company1name}.",
    "{name} contributed their expertise to {company1name}.",
    "{name} gained insights into the industry while working at {company1name}.",
    "{name} worked with clients and customers of {company1name}.",
    "{name} participated in projects that impacted {company1name}.",
    "{name} was part of the workforce driving innovation at {company1name}.",
    "{name} contributed their skills to the success of {company1name}.",
    "{name} worked at {company1name} and made a positive impact in their field.",
    "{name} was employed by {company1name}, a reputable company.",
    "{name} gained valuable experience at {company1name} in their role.",
    "{name} played a role in the success of {company1name}.",
    "{name} was involved in the day-to-day operations of {company1name}.",
    "{name} was an integral part of {company1name}'s team.",
    "{name} contributed to the growth and development of {company1name}.",
    "{name} made significant contributions to {company1name} during their tenure.",
    "{name} embraced the opportunities and challenges of working at {company1name}.",
    "{name} was a key asset to {company1name}'s success.",
    "{name} contributed to the achievements and milestones of {company1name}.",
    "{name} worked diligently at {company1name} to achieve their goals.",
]

# ── GPT-2 tokenizer with compact vocabulary (matching paper §A.3) ─────────────
# Paper limits to the ~3275 GPT-2 token IDs that actually appear in bioS text,
# so the embedding/LM head only covers tokens the dataset uses. This keeps the
# model size consistent with the paper's bits-per-parameter metric.

_TOKENIZER_CACHE = None    # raw GPT-2 tokenizer
_COMPACT_CACHE   = None    # (gpt2_id -> compact_id, compact_id -> gpt2_id, compact_vocab_size)

def _get_tokenizer():
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        from transformers import AutoTokenizer
        _TOKENIZER_CACHE = AutoTokenizer.from_pretrained("gpt2")
    return _TOKENIZER_CACHE


def _get_compact_vocab():
    """
    Build a compact token ID mapping by exhaustively enumerating all attribute values
    and sentence templates to collect every GPT-2 token ID that can appear in a bioS
    biography. Sampling is insufficient — rare attribute values may be missed, causing
    KeyErrors at training time and producing a vocab smaller than the paper's 3275.
    Returns (gpt2_to_compact, compact_to_gpt2, vocab_size).
    Cached after first call.
    """
    global _COMPACT_CACHE
    if _COMPACT_CACHE is not None:
        return _COMPACT_CACHE

    tok = _get_tokenizer()
    seen_ids: set = set()

    # Enumerate every attribute value that can appear verbatim in a biography.
    all_names = (
        [f"{f} {m} {l}" for f in FIRST_NAMES for m in MIDDLE_NAMES[:1] for l in LAST_NAMES[:1]]
        + [f"{f} {m} {l}" for f in FIRST_NAMES[:1] for m in MIDDLE_NAMES for l in LAST_NAMES[:1]]
        + [f"{f} {m} {l}" for f in FIRST_NAMES[:1] for m in MIDDLE_NAMES[:1] for l in LAST_NAMES]
    )
    birthdays = [
        f"{month} {day}, {year}"
        for month in BIRTH_MONTHS
        for day in BIRTH_DAYS
        for year in BIRTH_YEARS
    ]
    attr_pools = [
        FIRST_NAMES, MIDDLE_NAMES, LAST_NAMES,
        CITIES, UNIVERSITIES, MAJORS,
        EMPLOYERS, list(EMPLOYER_CITIES.values()),
        BIRTH_MONTHS, [str(d) for d in BIRTH_DAYS], [str(y) for y in BIRTH_YEARS],
        birthdays,
        ["He", "She"],
    ]
    for pool in attr_pools:
        for val in pool:
            for tid in tok.encode(" " + val):
                seen_ids.add(tid)

    # Enumerate every sentence template with a dummy placeholder so the surrounding
    # punctuation and whitespace tokens are all collected.
    dummy = "X"
    template_groups = [
        (_S1_BIRTHDAY,   {"name": dummy, "birthday": dummy}),
        (_S2_BIRTHCITY,  {"name": dummy, "birthcity": dummy}),
        (_S3_UNIVERSITY, {"name": dummy, "university": dummy}),
        (_S4_MAJOR,      {"name": dummy, "field": dummy}),
        (_S5_WORKCITY,   {"name": dummy, "company1city": dummy}),
        (_S6_EMPLOYER,   {"name": dummy, "company1name": dummy}),
    ]
    for templates, kwargs in template_groups:
        for tmpl in templates:
            for tid in tok.encode(" " + tmpl.format(**kwargs)):
                seen_ids.add(tid)

    seen_ids.add(tok.eos_token_id)
    # BPE cross-boundary tokens that only appear when company names with embedded
    # punctuation (e.g. "Caterpillar Inc.") are followed by "'s" or sentence-ending
    # punctuation. Cannot be recovered by enumerating parts independently.
    for s in ["..", ".,", ".'", "+.", "+,"]:
        for tid in tok.encode(s):
            seen_ids.add(tid)

    compact_to_gpt2 = sorted(seen_ids)
    gpt2_to_compact = {gpt2: compact for compact, gpt2 in enumerate(compact_to_gpt2)}
    _COMPACT_CACHE = (gpt2_to_compact, compact_to_gpt2, len(compact_to_gpt2))
    return _COMPACT_CACHE


def get_capo_vocab_size() -> int:
    """Return the compact vocabulary size (~3275, matching the paper)."""
    _, _, size = _get_compact_vocab()
    return size


# Expose vocab size — evaluated lazily so importing the module is fast.
# Call get_capo_vocab_size() to get the true compact size; use CAPO_VOCAB_SIZE
# only when the full GPT-2 vocab is required (e.g. loading old checkpoints).
CAPO_VOCAB_SIZE = 50257  # full GPT-2; kept for backward compat with old checkpoints


# ── Biography generation ───────────────────────────────────────────────────────

def _generate_attrs(rng, person_id):
    """Generate fixed biography attributes for one person."""
    first    = rng.choice(FIRST_NAMES)
    middle   = rng.choice(MIDDLE_NAMES)
    last     = rng.choice(LAST_NAMES)
    employer = rng.choice(EMPLOYERS)
    return {
        "first_name":   first,
        "middle_name":  middle,
        "last_name":    last,
        "id":           person_id,
        "birthmonth":   rng.choice(BIRTH_MONTHS),
        "birthday":     rng.choice(BIRTH_DAYS),
        "birthyear":    rng.choice(BIRTH_YEARS),
        "birthcity":    rng.choice(CITIES),
        "university":   rng.choice(UNIVERSITIES),
        "field":        rng.choice(MAJORS),
        "company1name": employer,
        "company1city": EMPLOYER_CITIES[employer],
    }


def _generate_text(attrs, rng):
    """
    Generate one paraphrase using per-sentence template sampling.
    Matches author's get_text_simple3 structure exactly:
      sentence 1: full name + birthday
      sentences 2-6: He/She pronoun + attribute
      sentences 5/6 (workcity/employer) are randomly permuted
    Each sentence is prefixed with " " then concatenated directly.
    """
    name     = f"{attrs['first_name']} {attrs['middle_name']} {attrs['last_name']}"
    pronoun  = "He" if attrs["id"] % 2 == 0 else "She"
    birthday = f"{attrs['birthmonth']} {attrs['birthday']}, {attrs['birthyear']}"

    s1 = " " + rng.choice(_S1_BIRTHDAY).format(name=name,    birthday=birthday)
    s2 = " " + rng.choice(_S2_BIRTHCITY).format(name=pronoun, birthcity=attrs["birthcity"])
    s3 = " " + rng.choice(_S3_UNIVERSITY).format(name=pronoun, university=attrs["university"])
    s4 = " " + rng.choice(_S4_MAJOR).format(name=pronoun,    field=attrs["field"])

    if rng.random() < 0.5:
        s5 = " " + rng.choice(_S5_WORKCITY).format(name=pronoun, company1city=attrs["company1city"])
        s6 = " " + rng.choice(_S6_EMPLOYER).format(name=pronoun, company1name=attrs["company1name"])
    else:
        s5 = " " + rng.choice(_S6_EMPLOYER).format(name=pronoun, company1name=attrs["company1name"])
        s6 = " " + rng.choice(_S5_WORKCITY).format(name=pronoun, company1city=attrs["company1city"])

    return s1 + s2 + s3 + s4 + s5 + s6


def generate_bio(rng, person_id=0):
    """Generate attributes and one paraphrase (kept for compatibility)."""
    attrs = _generate_attrs(rng, person_id)
    text  = _generate_text(attrs, rng)
    return text, attrs


# ── Dataset ────────────────────────────────────────────────────────────────────

class CapoDataset(IterableDataset):
    """
    Pretrains on N biographies with `exposures` paraphrases each.

    - Rank-sharded: each DDP rank sees a disjoint subset of (bio, exposure) pairs.
    - Finite: yields exactly N*exposures/world_size/num_workers items, then stops.
      Training ends naturally after one full pass (100 exposures by default).
    - GPT-2 tokenized to match the paper's bits-per-parameter measurement.
    """

    def __init__(self, N=50000, exposures=100, context_len=512, seed=42,
                 rank=0, world_size=1):
        super().__init__()
        self.N           = N
        self.exposures   = exposures
        self.context_len = context_len
        self.seed        = seed
        self.rank        = rank
        self.world_size  = world_size

        rng = random.Random(seed)
        self.attrs = [_generate_attrs(rng, i) for i in range(N)]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id   = worker_info.id          if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        all_pairs    = [(b, e) for b in range(self.N) for e in range(self.exposures)]
        rank_pairs   = all_pairs[self.rank::self.world_size]
        worker_pairs = rank_pairs[worker_id::num_workers]

        rng = random.Random(self.seed + self.rank * 1000 + worker_id)
        rng.shuffle(worker_pairs)

        tok    = _get_tokenizer()
        gpt2_to_compact, _, _ = _get_compact_vocab()
        eos_id = gpt2_to_compact[tok.eos_token_id]

        for bio_id, exp_idx in worker_pairs:
            exp_rng = random.Random(self.seed + bio_id * self.exposures + exp_idx)
            text    = _generate_text(self.attrs[bio_id], exp_rng)
            gpt2_ids = tok.encode(text)[: self.context_len]
            tokens   = [gpt2_to_compact[t] for t in gpt2_ids]
            tokens  += [eos_id] * (self.context_len - len(tokens))
            yield torch.tensor(tokens, dtype=torch.long)


def build_capo_dataset(N=50000, exposures=100, context_len=512, seed=42,
                       rank=0, world_size=1):
    return CapoDataset(N, exposures, context_len, seed=seed,
                       rank=rank, world_size=world_size)
