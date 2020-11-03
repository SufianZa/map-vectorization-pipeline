$(document).ready(function () {
    if ($('#diff-enlarge').is(':checked')) {
        $('#contours').show()
        $('#distance').hide()
    }

    if ($('#polysnap').is(':checked')) {
        $('#contours').hide()
        $('#distance').show()
    }

    $('#diff-enlarge').change(function (e) {
        if ($('#diff-enlarge').is(':checked')) {
            $('#contours').show()
            $('#distance').hide()
        }
    })

    $('#polysnap').change(function (e) {
        if ($('#polysnap').is(':checked')) {
            $('#contours').hide()
            $('#distance').show()
        }
    })
 $('#updateSensitivitySpan').text($('#updateSensitivity').val())
    $('#updateDouglasSpan').text($('#updateDouglas').val())
       $('#updateSensitivity').change(function (e) {
        $('#updateSensitivitySpan').text($(this).val())
    });
       $('#updateDouglas').change(function (e) {
        $('#updateDouglasSpan').text($(this).val())
    });

});