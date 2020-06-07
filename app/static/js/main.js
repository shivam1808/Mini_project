$(document).ready(function () {
    // Init
    $('#btn-heatmap').hide();
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('#prob').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#btn-heatmap').hide();
        $('#result').text('');
        $('#result').hide();
        $('#prob').hide().text('');
        $('#prob').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        // Show loading animation
        $(this).hide();
        $('.loader').show();
        $('#upload-file').hide()

        // Make prediction by calling api /predict
        $.ajax({
            dataType: "json",
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                var prediction = data.prediction;
                var probability =data.probability;
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#prob').fadeIn(600);
                $('#upload-file').hide()
                $('#result').text(' Covid-19:  ' + prediction);
                $('#prob').text(' Probability:  ' + probability);
                $('#btn-heatmap').show();
                console.log(data);
            },
        });
    });

});
