install.packages("httr")
install.packages("jsonlite")

library("httr")
library("jsonlite")


full_url1 <- "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/TK/TK1001/TK1001A/PersBilarDrivMedel"

json_fråga1 <- '{
  "query": [
    {
      "code": "Region",
      "selection": {
        "filter": "item",
        "values": [
          "00"
        ]
      }
    },
    {
      "code": "Drivmedel",
      "selection": {
        "filter": "item",
        "values": [
          "100",
          "110",
          "120",
          "130",
          "140",
          "150",
          "160",
          "190"
        ]
      }
    },
    {
      "code": "Tid",
      "selection": {
        "filter": "item",
        "values": [
          "2018M01",
          "2018M02",
          "2018M03",
          "2018M04",
          "2018M05",
          "2018M06",
          "2018M07",
          "2018M08",
          "2018M09",
          "2018M10",
          "2018M11",
          "2018M12",
          "2019M01",
          "2019M02",
          "2019M03",
          "2019M04",
          "2019M05",
          "2019M06",
          "2019M07",
          "2019M08",
          "2019M09",
          "2019M10",
          "2019M11",
          "2019M12",
          "2020M01",
          "2020M02",
          "2020M03",
          "2020M04",
          "2020M05",
          "2020M06",
          "2020M07",
          "2020M08",
          "2020M09",
          "2020M10",
          "2020M11",
          "2020M12",
          "2021M01",
          "2021M02",
          "2021M03",
          "2021M04",
          "2021M05",
          "2021M06",
          "2021M07",
          "2021M08",
          "2021M09",
          "2021M10",
          "2021M11",
          "2021M12",
          "2022M01",
          "2022M02",
          "2022M03",
          "2022M04",
          "2022M05",
          "2022M06",
          "2022M07",
          "2022M08",
          "2022M09",
          "2022M10",
          "2022M11",
          "2022M12"
        ]
      }
    }
  ],
  "response": {
    "format": "json"
  }
}'

# Send request
response1 <- POST(full_url1, body = json_fråga1, encode = "json", add_headers(`Content-Type` = "application/json"))

# Load data
NyregistreradePersonbilar <- content(response1, type = "text", encoding = "UTF-8")
NyregistreradePersonbilar1 <- fromJSON(NyregistreradePersonbilar)


# Extract data
Region <- unlist(lapply(NyregistreradePersonbilar1$data$key, "[[", 1))
Drivmedel <- unlist(lapply(NyregistreradePersonbilar1$data$key, "[[", 2))
Tid <- unlist(lapply(NyregistreradePersonbilar1$data$key, "[[", 3))
Antal <- as.numeric(replace(unlist(NyregistreradePersonbilar1$data$values), 
                            NyregistreradePersonbilar1$data$values == "..", NA))

# Create data frame
NyregistreradePersonbilar2 <- data.frame(
  Region = Region,
  Drivmedel = Drivmedel,
  Tid = Tid,
  Antal = Antal
)

# Add column
NyregistreradePersonbilar2$Datum <- as.Date(paste0(substr(NyregistreradePersonbilar2$Tid, 1, 4), "-",
                                                   substr(NyregistreradePersonbilar2$Tid, 6, 7), "-01"))


# Convert to Year
NyregistreradePersonbilar2 <- NyregistreradePersonbilar2[order(NyregistreradePersonbilar2$Datum), ]
NyregistreradePersonbilar2$År = format(NyregistreradePersonbilar2$Datum, "%Y")
NyregistreradePersonbilar3 <- aggregate(Antal ~ År, data = NyregistreradePersonbilar2, sum)


str(NyregistreradePersonbilar3)

#--------------------------------------------------------------

full_url2 <- "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/TK/TK1001/TK1001A/PersBilarA"

json_fråga2 <- '{
  "query": [
    {
      "code": "Region",
      "selection": {
        "filter": "item",
        "values": [
          "00"
        ]
      }
    },
    {
      "code": "Agarkategori",
      "selection": {
        "filter": "item",
        "values": [
          "000"
        ]
      }
    },
    {
      "code": "Tid",
      "selection": {
        "filter": "item",
        "values": [
          "2018",
          "2019",
          "2020",
          "2021",
          "2022"
        ]
      }
    }
  ],
  "response": {
    "format": "json"
  }
}'

# Send request
response2 <- POST(full_url2, body = json_fråga2, encode = "json", add_headers(`Content-Type` = "application/json"))

# Load data
PersonbilarTrafik <- content(response2, type = "text", encoding = "UTF-8")
PersonbilarTrafik1 <- fromJSON(PersonbilarTrafik)


# Extract data
Region <- unlist(lapply(PersonbilarTrafik1$data$key, "[[", 1))
Agarkategori <- unlist(lapply(PersonbilarTrafik1$data$key, "[[", 2))
Tid <- unlist(lapply(PersonbilarTrafik1$data$key, "[[", 3))
Antal <- as.numeric(replace(unlist(PersonbilarTrafik1$data$values), 
                            PersonbilarTrafik1$data$values == "..", NA))


# Create data frame
PersonbilarTrafik2 <- data.frame(
  Region = Region,
  Agarkategori = Agarkategori,
  Tid = Tid,
  Antal = Antal
)
str(PersonbilarTrafik2)

#--------------------------------------------------------------

# Plot Nyregistrerade personbilar (2018 - 2022)
ggplot(NyregistreradePersonbilar3, aes(x = År, y = Antal)) +
  geom_point(color = "red") +  
  labs(title = "Nyregistrerade personbilar (2018 - 2022)",
       x = "Year",
       y = "Nyregistrerade personbilar") +
  theme_minimal()


# Plot Personbilar i trafik
ggplot(PersonbilarTrafik2, aes(x = Tid, y = Antal)) +
  geom_point(color = "red") +  
  labs(title = "Personbilar i trafik (2018 - 2022)",
       x = "Year",
       y = "Personbilar i trafik") +
  theme_minimal()

