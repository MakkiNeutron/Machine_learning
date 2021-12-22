function getBathValue() {
  var uiBathrooms = document.getElementsByName("uiBathrooms");
  for(var i in uiBathrooms) {
    if(uiBathrooms[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1; // Invalid Value
}

function getBHKValue() {
  var uiBHK = document.getElementsByName("uiBHK");
  for(var i in uiBHK) {
    if(uiBHK[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1; // Invalid Value
}

function onClickedEstimatePrice() {
  console.log("Estimate price button clicked");
  var sqft = document.getElementById("uiSqft");
  var hometype = document.getElementById("uihometypes");
  var bedrooms = getBHKValue();
  var bathrooms = getBathValue();
  var location = document.getElementById("uiLocations");
  var estPrice = document.getElementById("uiEstimatedPrice");

  var url = "http://127.0.0.1:5000/predict_home_price"; //Use this if you are NOT using nginx which is first 7 tutorials
  //var url = "/api/predict_home_price"; // Use this if  you are using nginx. i.e tutorial 8 and onwards

  $.post(url, {
      area_sqft: parseFloat(sqft.value),
	  hometype: hometype,
      bedrooms: bedrooms,
      baths: bathrooms,
      location: location.value
  },function(data, status) {
      console.log(data.estimated_price);
      estPrice.innerHTML = "<h2>PKR " + data.estimated_price.toString() + " </h2>";
      console.log(status);
  });
}

function onPageLoad() {
  console.log( "document loaded" );
  var url_L = "http://127.0.0.1:5000/get_location_names"; // Use this if you are NOT using nginx which is first 7 tutorials
  var url_Ht = "http://127.0.0.1:5000/get_home_types"; // Use this if you are NOT using nginx which is first 7 tutorials
  
  
  //var url = "/api/get_location_names"; // Use this if  you are using nginx. i.e tutorial 8 and onwards
  $.get(url_L,function(data, status) {
      console.log("got response for get_location_names request");
      if(data) {
          var locations = data.locations;
          var uiLocations = document.getElementById("uiLocations");
          $('#uiLocations').empty();
          for(var i in locations) {
              var opt = new Option(locations[i]);
              $('#uiLocations').append(opt);
          }
      }
  });
  $.get(url_Ht,function(data, status) {
      console.log("got response for get_home_types request");
      if(data) {
          var hometype = data.home_types;
          var uihometypes = document.getElementById("uihometypes");
          $('#uihometypes').empty();
          for(var i in hometype) {
              var opt = new Option(hometype[i]);
              $('#uihometypes').append(opt);
          }
      }
  });
}

window.onload = onPageLoad;
