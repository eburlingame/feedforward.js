var x = "black", y = 10;
prevX = 0,
currX = 0,
prevY = 0,
currY = 0,
dot_flag = false;
function init() 
{
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext("2d");
    w = canvas.width;
    h = canvas.height;
    ctx.fillStyle = "#e4e4e2";
    ctx.clearRect(0, 0, w, h);

    canvas.addEventListener("mousemove", function (e) {
        findxy('move', e)
    }, false);
    canvas.addEventListener("mousedown", function (e) {
        findxy('down', e)
    }, false);
    canvas.addEventListener("mouseup", function (e) {
        findxy('up', e)
    }, false);
    canvas.addEventListener("mouseout", function (e) {
        findxy('out', e)
    }, false);
}

function draw() 
{
    ctx.fillStyle = x;
    ctx.fillRect(currX, currY, y, y);
}

function erase() 
{
    ctx.clearRect(0, 0, w, h);
}

function save() 
{
    document.getElementById("canvasimg").style.border = "2px solid";
    var dataURL = canvas.toDataURL();
    document.getElementById("canvasimg").src = dataURL;
    document.getElementById("canvasimg").style.display = "inline";
}

function findxy(res, e) 
{
    if (res == 'down') 
    {
        prevX = currX;
        prevY = currY;
        currX = e.clientX - canvas.offsetLeft;
        currY = e.clientY - canvas.offsetTop;

        flag = true;
        dot_flag = true;
        if (dot_flag) 
        {
            ctx.beginPath();
            ctx.fillStyle = x;
            ctx.fillRect(currX, currY, y, y);
            ctx.closePath();
            dot_flag = false;
        }
    }
    if (res == 'up' || res == "out") 
    {
        flag = false;
    }
    if (res == 'move') 
    {
        if (flag) 
        {
            prevX = currX;
            prevY = currY;
            currX = e.clientX - canvas.offsetLeft;
            currY = e.clientY - canvas.offsetTop;
            draw();
        }
    }
}