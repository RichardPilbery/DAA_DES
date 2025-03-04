from scour.scour import start as scour, parse_args as scour_args, getInOut as scour_io

def scour_svg(infilename,outfilename):
    options = scour_args()
    options.infilename = infilename
    options.outfilename = outfilename
    (input, output) = scour_io(options)
    scour(options, input, output)

if __name__ == '__main__':
    infilename = 'app/assets/daa-logo.svg'
    outfilename = 'app/assets/daa-logo_scoured.svg'
    scour_svg(infilename,outfilename)
